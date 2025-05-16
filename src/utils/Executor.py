from utils.config_parser import Config
from utils.logging import TimeLogger, DataLogger
from dataloaders.dataloader import get_floodnet_dataloader
from utils.transforms import classification_image_tf, segmentation_image_tf, segmentation_mask_tf
from utils.train import train_model
from utils.evaluate import evaluate_model
from collections import defaultdict
import csv
import json
import os
import pathlib

SAVED_MODELS_DIR = "saved_models/"

class Executor():
    def __init__(self, device: str):
        self.device = device
        self.time_logger = TimeLogger()
        self.data_logger = DataLogger()

    def execute_config(self, config_path: str, device: str, verbose: bool):
        config = Config(config_path)
        print(f"- Model name: {config.model.name}")
        print(f"- Loss: {config.loss}")
        print(f"- Optimizer: {config.optimizer}")
        print(f"- Metrics: {config.metrics}")

        train_data, val_data, _ = self.__get_data(config=config)

        for epoch in range(config.epochs):
            if verbose:
                print(f"-------------- Epoch {epoch+1} --------------")
            train_model(train_data, config, self.time_logger, self.data_logger, device, verbose)
            if verbose:
                print("---------------------------------------------")
            evaluate_model(val_data, config, self.time_logger, self.data_logger, device, verbose)

            if verbose:
                print("----- Epoch times -----")
                print("Training: ", end="")
                self.time_logger.print_log(f"{config.config_name}_train_time")
                print("Validation: ", end="")
                self.time_logger.print_log(f"{config.config_name}_val_time")
                print("----- Epoch times average -----")
                print("Training: ", end="")
                self.time_logger.print_log_avg(f"{config.config_name}_train_time")
                print("Validation: ", end="")
                self.time_logger.print_log_avg(f"{config.config_name}_val_time")
                print("----- Logged data -----")
                print(json.dumps(self.data_logger.logs, indent=4))

        path = pathlib.Path(SAVED_MODELS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        config.model.save_model(os.path.join(SAVED_MODELS_DIR, f"{config.config_name}.pth"))

        del config.model

        if verbose:
            print("Done!")

    def save_logs(self, file_path):
        logs = self.__combine_logs()
        fieldnames = logs.keys()
        rows = zip(*logs.values())

        with open(file_path, 'w', newline='') as f:
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

    def __get_data(self, config):
        batch_size = config.batch_size
        transforms = self.__get_data_transforms(config.task)

        train_inputs, train_targets = config.train_data_paths
        train_data = get_floodnet_dataloader(inputs_path=train_inputs,
                                             targets_path=train_targets,
                                             transforms=transforms,
                                             batch_size=batch_size,
                                             shuffle=True)

        val_inputs, val_targets = config.val_data_paths
        val_data = get_floodnet_dataloader(inputs_path=val_inputs,
                                           targets_path=val_targets,
                                           transforms=transforms,
                                           batch_size=batch_size,
                                           shuffle=False)

        test_inputs, test_targets = config.test_data_paths
        test_data = get_floodnet_dataloader(inputs_path=test_inputs,
                                            targets_path=test_targets,
                                            transforms=transforms,
                                            batch_size=batch_size,
                                            shuffle=False)
        
        return train_data, val_data, test_data
        
    def __get_data_transforms(self, task):
        if task == "classification":
            return [classification_image_tf]
        if task == "segmentation":
            return [segmentation_image_tf, segmentation_mask_tf]
