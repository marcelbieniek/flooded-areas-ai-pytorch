import torch
from torch import nn

def bce_with_logits():
    """Binary Cross Entropy with Logits"""
    return nn.BCEWithLogitsLoss()

def cross_entropy():
    return nn.CrossEntropyLoss()

def binary_dice():
    return binary_dice_loss

def multiclass_dice():
    return multiclass_dice_loss

def binary_dice_loss(inputs, targets, smooth=1e-6):
    """Sorensen-Dice Loss for binary classification"""
    # Apply sigmoid to get probabilities from logits if not already done
    inputs = torch.sigmoid(inputs)

    # Flatten the tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Calculate Dice coefficient
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

def multiclass_dice_loss(inputs, targets, smooth=1e-6):
    """Sorensen-Dice Loss for multiclass semantic segmentation (>=2 classes)"""
    # Apply softmax to get probabilities
    inputs = nn.functional.softmax(inputs, dim=1)
    num_classes = inputs.shape[1]

    # One-hot encode the target
    targets_one_hot = nn.functional.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

    # Flatten predictions and targets
    inputs = inputs.contiguous().view(inputs.shape[0], num_classes, -1)
    targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], num_classes, -1)

    # Compute Dice score
    intersection = (inputs * targets_one_hot).sum(-1)
    union = inputs.sum(-1) + targets_one_hot.sum(-1)

    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score

    # Average over batch and classes
    return dice_loss.mean()
