import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3():
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        self.name = "DeepLabV3"
        self.num_classes = num_classes

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=pretrained)
        self.model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, X):
        return self.model(X)["out"]

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

def deeplabv3(num_classes, pretrained):
    return DeepLabV3(num_classes, pretrained)
