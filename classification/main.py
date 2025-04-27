import torch
import os
from utils.config_parser import Config
from dataloader.dataloader import *
from train import train_model
from evaluate import test_model
from utils.logger import TimeLogger, DataLogger
from utils.utils import plot_predictions
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

current_cuda_device = None
print(f"is cuda available: {torch.cuda.is_available()}")
print(f"cuda device count: {torch.cuda.device_count()}")
current_cuda_device = torch.cuda.current_device()
print(f"current cuda device: {current_cuda_device}")
print(f"current cuda device name: {torch.cuda.get_device_name(current_cuda_device)}")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

config = Config("config/classification/inceptionnetv3_config.yaml")
print(config.model)
print(config.loss)
print(config.optimizer)
print(config.metrics)
print(config.metrics_names)

timer = TimeLogger()
logger = DataLogger()
train_data, val_data = get_dataloaders(config.config["train"]["batch_size"])
test_data = get_test_dataloader(config.config["train"]["batch_size"])

epochs = config.config["train"]["epochs"]

for epoch in range(epochs):
    print(f"-------------- Epoch {epoch+1} --------------")
    train_model(train_data, config, timer, logger, device)
    test_model(val_data, config, timer, logger, device)
    print("----- Epoch times:")
    timer.print_log(f"{config.model.name}_train")
    timer.print_log(f"{config.model.name}_val")
    print("----- Epoch times avg:")
    timer.print_log_avg(f"{config.model.name}_train")
    timer.print_log_avg(f"{config.model.name}_val")
    print("----- Logged data:")
    print(logger.logs)
print("Done!")

config.model.save_model("inception.pth")

# Loss
plt.figure(figsize=(10, 5))
plt.plot(logger.logs["InceptionNetV3_train_loss"], label='Train Loss')
plt.plot(logger.logs["InceptionNetV3_val_loss"], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("inception_loss")

# config.model.load_model("inception.pth")
plot_predictions(config.model, test_data, {0:"non-flooded", 1:"flooded"})
