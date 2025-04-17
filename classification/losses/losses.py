import torch
from torch import nn

def bce_with_logits():
    return nn.BCEWithLogitsLoss()

def dice():
    """Sorensen-Dice Loss"""
    return sorensen_dice

# TODO
def sorensen_dice(preds, targets):
    print(f"sdl received {preds.size()}, {targets.size()}")
