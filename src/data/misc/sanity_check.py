import os
from pathlib import Path
import numpy as np
from PIL import Image


# when set to True, the sanity check will also resize any image that's not of expected size, to the excted size
# otherwise raise AssertionError on first image that's not of expected size
RESIZE = False


# set variables
root_dir = Path("FloodNet_dataset/")
data_types = ["train", "test", "val"]
expected_size = (4000, 3000) # (width, height), in pixels

for data_type in data_types:
    img_dir_path = Path(f"{root_dir}/{data_type}/image/")
    label_dir_path = os.path.abspath(f"{root_dir}/{data_type}/label/")

    for label_name in os.listdir(label_dir_path):
        print(f"--------------- {label_name} ---------------")

        label_full_path = os.path.join(label_dir_path, label_name)
        label = Image.open(label_full_path)

        label_data = np.array(label)
        values, counts = np.unique(label_data, return_counts=True)
        features = dict(zip(values, counts))
        width, height = label.size
        total_pixels = width * height

        if (width, height) != expected_size:
            if RESIZE:
                image_name = label_name.replace("_lab.png", ".jpg")
                image_full_path = os.path.join(img_dir_path, image_name)
                image = Image.open(image_full_path)

                image = image.resize(expected_size)
                label = label.resize(expected_size)
                image.save(image_full_path)
                label.save(label_full_path)
            else:
                raise AssertionError(f"Error, wrong size! Label {label_name} has actual size of: {width} x {height}")
