import os
import sys

from torch.utils.data import DataLoader

modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)

from data.FloodNet import FloodNetClassification

def get_dataloader(img_dir, labels_file, transform, batch_size, shuffle):
    data = FloodNetClassification(img_dir=img_dir, annotations_file=labels_file, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
