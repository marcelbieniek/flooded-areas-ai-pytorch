import timm
from torch import nn

class Xception():
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        self.num_classes = num_classes
        self.model = timm.create_model('xception', pretrained=pretrained)

        # set final number of classes
        self.model.fc = nn.Linear(self.model.num_features, num_classes)

    def train(self, X):
        return self.model(X)
    
    def calculate_loss(self, loss_fn, model_outputs, y):
            return loss_fn(model_outputs, y)
    
    def parameters(self):
        return self.model.parameters()
    
    def train_mode(self):
        self.model.train()

    def move_to_device(self, device):
        self.model = self.model.to(device)

def xception(num_classes, pretrained):
    return Xception(num_classes, pretrained)
