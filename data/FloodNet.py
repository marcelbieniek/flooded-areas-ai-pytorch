import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image

class FloodNetClassification(Dataset):
    def __init__(self, img_dir: str, annotations_file: str, classes: dict = None, transform=None):
        """
        Args:
            img_dir (string): Path to root directory with all the images.
            annotations_file (string): Path to the CSV file with annotations (image paths, labels).
            classes (dict, optional): Optional dictionary describing classes in the dataset, in format {label_value (int): "class_name"}
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = img_dir
        self.data = pd.read_csv(annotations_file)
        self.transform = transform
        if classes:
            self.classes = dict(sorted(classes.items(), key=lambda item: item[0]))
        else:
            self.classes = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = Image.open(img_path)
        label = self.data.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_dict(self):
        return self.classes

    def get_class_names(self):
        return list(self.classes.values()) if self.classes else None


class FloodNetSegmentation(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, classes: dict = None, image_transform=None, mask_transform=None):
        """
        Args:
            img_dir (string): Path to root directory with all the images.
            mask_dir (string): Path to root directory with all the image masks.
            classes (dict, optional): Optional dictionary describing classes in the dataset, in format {label_value (int): "class_name"}
            image_transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a sample mask.
        """
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(img_dir))
        if classes:
            self.classes = dict(sorted(classes.items(), key=lambda item: item[0]))
        else:
            self.classes = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        label_name = image_name.replace(".jpg", "_lab.png")
        mask = Image.open(os.path.join(self.mask_dir, label_name))

        # if index == 674 or index == 675:
        #     print(label_name)
        
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def get_class_dict(self):
        return self.classes

    def get_class_names(self):
        return list(self.classes.values()) if self.classes else None
