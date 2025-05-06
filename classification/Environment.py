import argparse
import os
import torch
import sys

ROOT_CONFIG_DIR = "config"

class Environment():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="Flooded Areas AI",
            description="An evaluation workflow for classification and semantic segmentation neural network models."
        )

        self.parser.add_argument("-c", "--cuda-device", type=int, default=0, help="Index of CUDA device to compute on, if available (default=0).")
        self.parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print additional information during execution.")
        self.parser.add_argument("-r", "--run", type=str, default="all", help="""Specify which model configuration to run.
                                 All configuration files are expected to be in 'config' directory, positioned at the root of the project.
                                 Allowed values are:
                                 - all -- Run all configs found in the 'config' directory.
                                 - file_name -- Path to YAML file to use as config (requires file extension .yaml or .yml; path should begin from first level inside 'config' directory).
                                 - subdir_name -- Name of subdirectory inside the 'config' directory. All configs from this and further subdirectories will be run (can be used for grouping configs eg. run all segmentation models).
                                 """)
        self.args = self.parser.parse_args()

        self.device = None
        self.configs = []

    def init(self):
        self.__set_cuda_environment()
        self.__find_configs()
        self.__validate_found_configs()

    def run(self):
        # for each config:
        #   create Runner <- Runner class
        #   run model <- Runner class
        #   save results <- Runner class, path to save created and provided by Environment
        pass

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

        if self.args.verbose:
            print("----- Model configuration files found -----")
            for config_path in self.configs:
                print(config_path)

    def __set_cuda_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_device)

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if self.args.verbose:
            print("----- Computation device -----")
            print(f"Using {self.device} for computations")
            if self.device == "cuda":
                print(f"Using cuda device: {self.args.cuda_device}")
                print(f"Current cuda device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
                print(f"Cuda device count: {torch.cuda.device_count()}")
