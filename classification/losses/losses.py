import torch
from torch import nn

def bce_with_logits():
    return nn.BCEWithLogitsLoss()
