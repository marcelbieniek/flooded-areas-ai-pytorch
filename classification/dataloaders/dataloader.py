import os
import sys

from torch.utils.data import DataLoader

# modules_path = os.path.abspath("..")
# if modules_path not in sys.path:
#     sys.path.append(modules_path)

from data.FloodNet import FloodNetClassification, FloodNetSegmentation

def get_dataloader(inputs_path: str, targets_path: str, transforms: tuple, batch_size: int, shuffle: bool):
    """
    Args:
        inputs_path (string): Path to directory containing data inputs.
        targets_path (string): Path to CSV file (format: path_to_image, class) for image classification task, or directory with segmentation masks for image segmentation task.
        transforms (tuple): Transforms to be applied on inputs and targets. If targets is a CSV file then no target transform is applied.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
    """
    if not os.path.isdir(inputs_path):
        raise NotADirectoryError(f"Argument 'inputs_path' expected a directory, but got: '{inputs_path}'")

    data = None
    if targets_path.endswith(".csv"):
        data = FloodNetClassification(img_dir=inputs_path,
                                      annotations_file=targets_path,
                                      transform=transforms[0]
                                      )
    elif os.path.isdir(targets_path):
        data = FloodNetSegmentation(img_dir=inputs_path,
                                    mask_dir=targets_path,
                                    image_transform=transforms[0],
                                    mask_transform=transforms[1]
                                    )
    else:
        raise TypeError(f"Argument 'targets_path' expected a path to a CSV file or a directory containing segmentation masks, got '{targets_path}'")

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
