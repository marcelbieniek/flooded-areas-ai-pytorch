import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FloodNetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the CSV file with annotations (image paths, labels)
            img_dir (string): Path to root directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = pd.read_csv(annotations_file)
        self.root_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = Image.open(img_path)
        label = self.data.iloc[index, 1]

        if self.transform:
            image = self.transform(image)
        
        return image, label
