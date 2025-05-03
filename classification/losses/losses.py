import torch
from torch import nn

def bce_with_logits():
    """Binary Cross Entropy with Logits"""
    return nn.BCEWithLogitsLoss()

def dice():
    """Sorensen-Dice Loss"""
    return sorensen_dice

def sorensen_dice(preds, targets, smooth=1e-6):
    # Apply sigmoid to get probabilities from logits if not already done
    preds = torch.sigmoid(preds)

    # Flatten the tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate Dice coefficient
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

def cross_entropy():
    return nn.CrossEntropyLoss()
