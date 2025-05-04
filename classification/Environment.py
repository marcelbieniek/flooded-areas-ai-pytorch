import argparse
import os
import torch
import sys

class Environment():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="Flooded Areas AI",
            description="An evaluation workflow for classification and semantic segmentation neural network models."
        )

        self.parser.add_argument("-c", "--cuda-device", type=int, default=0, help="Index of CUDA device to compute on, if available (default=0).")
        self.parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Print additional information during execution.")
        self.parser.add_argument("-r", "--run", type=str, default="all", help="""Specify which model to run.
                                 All configuration files are expected to be in 'config' directory, positioned at the root of the project.
                                 Allowed values are:
                                 - all -- Run all configs found in the 'config' directory.
                                 - file_name -- Name of YAML file to use as config.
                                 - subdir_name -- Name of subdirectory inside the 'config' directory. All configs from this subdirectory will be run (can be used for grouping configs eg. run all segmentation models).
                                 """)
        self.args = self.parser.parse_args()

        self.device = None
        self.__set_cuda_environment()

        self.configs = []
        self.__find_configs()

    def print_root_path(self):
        print(sys.path[0])

    def run(self):
        pass

    def __find_configs(self):
        for root, dirs, files in os.walk("config"):
            if self.args.run == "all":
                for f in files:
                    self.configs.append(os.path.join(root, f))
                continue

            # if self.args.run in dirs:
            #     for sub_root, sub_dir, sub_files

        print(self.configs)

    def __set_cuda_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_device)

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        current_cuda_device = torch.cuda.current_device()
        if self.args.verbose and self.device == "cuda":
            print(f"current cuda device: {current_cuda_device}")
            print(f"current cuda device name: {torch.cuda.get_device_name(current_cuda_device)}")
            print(f"cuda device count: {torch.cuda.device_count()}")
            print(f"Using {self.device} device")
