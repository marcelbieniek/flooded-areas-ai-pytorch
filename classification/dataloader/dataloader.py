from pathlib import Path
import os
import sys
modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)
from data.FloodNet import FloodNetDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    # Create train dataset
    train_images_dir = Path("../data/FloodNet_dataset/train/image")
    train_csv_file = Path("../data/flood_train_rel_paths.csv")
    train_data = FloodNetDataset(annotations_file=train_csv_file, img_dir=train_images_dir, transform=transform)

    # Create test dataset
    test_images_dir = Path("../data/FloodNet_dataset/test/image")
    test_csv_file = Path("../data/flood_test_rel_paths.csv")
    test_data = FloodNetDataset(annotations_file=test_csv_file, img_dir=test_images_dir, transform=transform)

    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
