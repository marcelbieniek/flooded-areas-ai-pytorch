import torch
from torchvision.transforms import functional as TF
from PIL import Image
import os

# check = torch.load("masks_337.pt")

# print(torch.unique(check))


root_dir = "../data/FloodNet_dataset/"
data_types = ["train", "test", "val"]
test_tensor = torch.tensor([0,1,2,3,4,5,6,7,8,9])

invalid_labels = []

for data_type in data_types:
    label_dir_path = os.path.abspath(f"{root_dir}/{data_type}/label/")

    for label_name in os.listdir(label_dir_path):
        print(f"--------------- {label_name} ---------------")

        label_full_path = os.path.join(label_dir_path, label_name)
        label = Image.open(label_full_path)

        mask = TF.pil_to_tensor(label).long()
        mask_unique = torch.unique(mask)

        # Check for values outside of expected range
        outside_range = (mask_unique < torch.min(test_tensor).item()) | (mask_unique > torch.max(test_tensor).item())

        # If any value is outside the range
        if outside_range.any():
            invalid_labels.append(label_name)
            print(f"{label_name} --- Values outside expected range: {mask_unique[outside_range]}")

print(f"Invalid labels: {invalid_labels}")
