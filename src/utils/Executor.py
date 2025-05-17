from utils.Config import Config
from utils.logging import TimeLogger, DataLogger
from dataloaders.dataloader import get_floodnet_dataloader
from utils.transforms import classification_image_tf, segmentation_image_tf, segmentation_mask_tf
from utils.train import train_model
from utils.evaluate import evaluate_model
from collections import defaultdict
import csv
import json
import pathlib

class Executor():
    def __init__(self, config_path: str, device: str, verbose: bool = False):
        self.config = Config(config_path)
        self.device = device
        self.verbose = verbose
        self.time_logger = TimeLogger()
        self.data_logger = DataLogger()

    def execute_config(self):
        if self.verbose:
            print(f"- Model name: {self.config.model.name}")
            print(f"- Loss: {self.config.loss}")
            print(f"- Optimizer: {self.config.optimizer}")
            print(f"- Metrics: {self.config.metrics}")

        train_data, val_data, _ = self.__get_data()

        for epoch in range(self.config.epochs):
            if self.verbose:
                print(f"-------------- Epoch {epoch+1} --------------")
            train_model(train_data, self.config, self.time_logger, self.data_logger, self.device, self.verbose)
            if self.verbose:
                print("---------------------------------------------")
            evaluate_model(val_data, self.config, self.time_logger, self.data_logger, self.device, self.verbose)

            if self.verbose:
                print("----- Epoch times -----")
                print("Training: ", end="")
                self.time_logger.print_log(f"{self.config.config_name}_train_time")
                print("Validation: ", end="")
                self.time_logger.print_log(f"{self.config.config_name}_val_time")
                print("----- Epoch times average -----")
                print("Training: ", end="")
                self.time_logger.print_log_avg(f"{self.config.config_name}_train_time")
                print("Validation: ", end="")
                self.time_logger.print_log_avg(f"{self.config.config_name}_val_time")
                print("----- Logged data -----")
                print(json.dumps(self.data_logger.logs, indent=4))

        if self.verbose:
            print("Done!")
    
    def save_model(self, models_dir: pathlib.Path):
        self.config.model.save_model(models_dir / f"{self.config.config_name}.pth")

    def save_logs(self, logs_dir: pathlib.Path):
        logs = self.__combine_logs()
        fieldnames = logs.keys()
        rows = zip(*logs.values())

        with open(logs_dir / f"{self.config.config_name}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(rows)

    def __combine_logs(self):
        combined = defaultdict(list)

        # Add items from dict1
        for k, v in self.time_logger.logs.items():
            combined[k].extend(v)

        # Add items from dict2
        for k, v in self.data_logger.logs.items():
            combined[k].extend(v)

        # Convert back to regular dict if desired
        return dict(combined)

    def __get_data(self):
        batch_size = self.config.batch_size
        transforms = self.__get_data_transforms()

        train_inputs, train_targets = self.config.train_data_paths
        train_data = get_floodnet_dataloader(inputs_path=train_inputs,
                                             targets_path=train_targets,
                                             transforms=transforms,
                                             batch_size=batch_size,
                                             shuffle=True)

        val_inputs, val_targets = self.config.val_data_paths
        val_data = get_floodnet_dataloader(inputs_path=val_inputs,
                                           targets_path=val_targets,
                                           transforms=transforms,
                                           batch_size=batch_size,
                                           shuffle=False)

        test_inputs, test_targets = self.config.test_data_paths
        test_data = get_floodnet_dataloader(inputs_path=test_inputs,
                                            targets_path=test_targets,
                                            transforms=transforms,
                                            batch_size=batch_size,
                                            shuffle=False)
        
        return train_data, val_data, test_data
        
    def __get_data_transforms(self):
        task = self.config.task
        if task == "classification":
            return [classification_image_tf]
        if task == "segmentation":
            return [segmentation_image_tf, segmentation_mask_tf]
