import os
import sys

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)
from data.FloodNet import FloodNetSegmentation

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

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
    #                      std=[0.229, 0.224, 0.225])
])

dataset = FloodNetSegmentation(img_dir="../data/FloodNet_dataset/train/image",
                               mask_dir="../data/FloodNet_dataset/train/label",
                               transform=transform
                               )

train_data = DataLoader(dataset, batch_size=1, shuffle=False)

for X, y in train_data:
    print(X)
    print(y)
