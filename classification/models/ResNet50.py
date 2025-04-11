import torch
from torch import nn

class ResNet50():
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        self.num_classes = num_classes
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)

        # set final number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def train(self, X):
        return self.model(X)
    
    def calculate_loss(self, loss_fn, model_outputs, y):
            return loss_fn(model_outputs, y)
