import os
import json

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.config_parser import Config
from dataloaders.dataloader import get_dataloader
from train import train_model
from evaluate import test_model
from utils.logger import TimeLogger, DataLogger
from utils.utils import plot_predictions

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
epochs = config.epochs
batch_size = config.batch_size

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
    #                      std=[0.229, 0.224, 0.225])
])

train_data = get_dataloader(img_dir="../data/FloodNet_dataset/train/image",
                            labels_file="../data/flood_train_rel_paths.csv",
                            transform=transform,
                            batch_size=batch_size,
                            shuffle=True)

val_data = get_dataloader(img_dir="../data/FloodNet_dataset/val/image",
                          labels_file="../data/flood_val_rel_paths.csv",
                          transform=transform,
                          batch_size=batch_size,
                          shuffle=False)

test_data = get_dataloader(img_dir="../data/FloodNet_dataset/test/image",
                           labels_file="../data/flood_test_rel_paths.csv",
                           transform=transform,
                           batch_size=batch_size,
                           shuffle=False)

timer = TimeLogger()
logger = DataLogger()

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

# plot loss
plt.figure(figsize=(10, 5))
plt.plot(logger.logs["InceptionNetV3_train_loss"], label='Train Loss')
plt.plot(logger.logs["InceptionNetV3_val_loss"], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("inception_loss")

# inferencing on trained model
# config.model.load_model("inception.pth")
# with open("../data/classification_classes.json", "r") as f:
#     classes = json.load(f)
#     classes = {int(k): v for k, v in classes.items()}
#     print(classes)
# plot_predictions(config.model, test_data, classes)
