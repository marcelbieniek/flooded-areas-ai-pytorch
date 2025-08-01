import argparse
import os
import torch
import pathlib
from datetime import datetime
import time

from utils.Executor import Executor
from utils.TestSetup import TestSetup

ROOT_CONFIG_DIR = "configs"
RESULT_DIR = "results"

OPT_ARG_DEFAULT_VALUE = "NO_VALUE"

class Environment():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="Flooded Areas AI",
            description="An evaluation workflow for classification and semantic segmentation neural network models."
        )

        self.parser.add_argument("-c", "--cuda-device", type=int, default=0, help="Index of CUDA device to compute on, if available (default=0).")
        self.parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print additional information during execution.")
        self.parser.add_argument("-r", "--run", type=str, default="all", help="""Specify which model configuration to run.
                                 All configuration files are expected to be in 'configs' directory, positioned at the root of the project.
                                 Allowed values are:
                                 - all -- Run all configs found in the 'configs' directory.
                                 - file_name -- Path to YAML file to use as config (requires file extension .yaml or .yml; path should begin from first level inside 'configs' directory).
                                 - subdir_name -- Name of subdirectory inside the 'configs' directory. All configs from this and further subdirectories will be run (can be used for grouping configs eg. run all segmentation models).
                                 """)
        self.parser.add_argument("-l", "--logs", action="store_true", default=False, help="Collect logs and models to files.")
        self.parser.add_argument("-t", "--test", nargs='?', const=OPT_ARG_DEFAULT_VALUE, help="Enable model testing. If ")
        self.args = self.parser.parse_args()

        self.verbose = False
        self.logging = False
        self.device = None
        self.test_setup = TestSetup.NONE
        self.configs = []

    def init(self):
        if self.args.verbose:
            self.verbose = True
        self.__set_cuda_environment()
        if self.args.test:
            self.__validate_testing_setup()
        self.__find_configs()
        self.__validate_found_configs()

        if self.args.logs or self.logging:
            self.logging = True
            result_path = pathlib.Path(RESULT_DIR)
            result_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        start_time = time.time()
        run_dir = pathlib.Path(RESULT_DIR) / datetime.now().strftime("%Y_%m_%d-%H_%M")
        logs_dir = run_dir / "logs"
        models_dir = run_dir / "models"

        if self.logging:
            run_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)

            if self.verbose:
                print(f"----- Results will be saved to: {run_dir} -----")

        for config in self.configs:
            if self.verbose:
                print(f"----- Starting execution of config: {config} -----")

            executor = None
            if self.test_setup == TestSetup.AFTER_TRAINING:
                executor = Executor(config, self.device, self.verbose)
            elif self.test_setup == TestSetup.TESTING_ONLY:
                executor = Executor(config, self.device, self.verbose, self.test_setup, self.args.test)
            if executor == None:
                raise ValueError("Error occurred! Could not set up config execution")
            executor.execute_config()

            if self.logging:
                executor.save_logs(logs_dir)
                executor.save_model(models_dir)
                with open(run_dir / "configs.txt", "a") as f:
                    f.write(f"{config}\n")

            if self.verbose:
                print(f"----- Finished executing config on: {datetime.now()} -----")

        if self.verbose:
            print(f"----- Run took {time.time() - start_time}s")

    def __find_configs(self):
        arg = self.args.run
        if arg == "all":
            self.configs = self.__get_all_configs_in_dir(ROOT_CONFIG_DIR)
            return

        if arg.endswith((".yaml", ".yml")):
            self.configs.append(os.path.join(ROOT_CONFIG_DIR, arg))
            return

        for root, dirs, _ in os.walk(ROOT_CONFIG_DIR):
            for d in dirs:
                if arg in os.path.join(root, d):
                    self.configs = self.__get_all_configs_in_dir(os.path.join(ROOT_CONFIG_DIR, arg))
                    return

    def __get_all_configs_in_dir(self, root_dir):
        configs = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(('.yaml', '.yml')):
                    configs.append(os.path.join(root, f))
        return configs

    def __validate_found_configs(self):
        if not self.configs:
            raise ValueError(f"No model configuration file(s) found under given argument: '{self.args.run}'")

        for config_path in self.configs:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model configuration file '{config_path}' does not exist")

        if self.verbose:
            print("----- Model configuration files found -----")
            for config_path in self.configs:
                print(config_path)
            print("-------------------------------------------")

    def __validate_testing_setup(self):
        test_setup_arg = self.args.test
        if not test_setup_arg:
            return

        if test_setup_arg == OPT_ARG_DEFAULT_VALUE:
            self.test_setup = TestSetup.AFTER_TRAINING
            self.logging = True # need to enable logging to save model
            if self.verbose:
                print("----- Logging enabled to save model for testing (mandatory)")
                print("----- Model testing will be performed after training")
            return

        self.test_setup = TestSetup.TESTING_ONLY
        run_plan_arg = self.args.run
        if not run_plan_arg.endswith((".yaml", ".yml")):
            raise ValueError("Invalid or no configuration file provided for model testing.")
        self.__validate_model_path()
        if self.verbose:
            print(f"----- Testing model '{self.args.test}'...")

    def __validate_model_path(self):
        model_path = pathlib.Path(self.args.test)
        if not model_path.exists():
            raise FileNotFoundError(f"No such model file found: {model_path}")
        elif model_path.suffix not in [".pt", ".pth"]:
            raise ValueError(f"Invalid model file extension: {model_path.suffix}. Expected '.pt' or '.pth'.")

    def __set_cuda_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_device)

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if self.verbose:
            print("----- Computation device -----")
            print(f"Using {self.device} for computations")
            if self.device == "cuda":
                print(f"Using cuda device: {self.args.cuda_device}")
                print(f"Current cuda device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
                print(f"Cuda device count: {torch.cuda.device_count()}")
