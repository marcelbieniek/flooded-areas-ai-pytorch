import torch
from torch import nn

class ResNet50():
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        self.name = "ResNet50"
        self.num_classes = num_classes
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)

        # set final number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, X):
        return self.model(X)

    def calculate_loss(self, loss_fn, model_outputs, y):
            return loss_fn(model_outputs, y)

    def parameters(self):
        return self.model.parameters()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def move_to_device(self, device):
        self.model = self.model.to(device)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

def resnet50(num_classes, pretrained):
    return ResNet50(num_classes, pretrained)
