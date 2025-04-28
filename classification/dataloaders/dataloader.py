import os
import sys
from data.FloodNet import FloodNetDataset
from torch.utils.data import DataLoader

modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)

def get_dataloader(img_dir, labels_file, transform, batch_size, shuffle):
    data = FloodNetDataset(annotations_file=labels_file, img_dir=img_dir, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
