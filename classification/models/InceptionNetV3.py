import torch
from torch import nn

class InceptionNetV3():
    def __init__(self, num_classes: int = 1, pretrained: bool = True, aux_logits: bool = True):
        self.name = "InceptionNetV3"
        self.num_classes = num_classes
        self.aux_logits = aux_logits
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained, aux_logits=True) # aux_logits always true here, because of a bug in InceptionNetV3 implementation, argument still has desired effect when calculating loss

        # set final number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if self.model.aux_logits:
            # print("replacing aux classifier")
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)

    def train(self, X):
        return self.model(X)
    
    def calculate_loss(self, loss_fn, model_outputs, y):
        if self.aux_logits:
            outputs, aux_outputs = model_outputs
            loss1 = loss_fn(outputs, y)
            loss2 = loss_fn(aux_outputs, y)
            return loss1 + 0.4 * loss2
        else:
            return loss_fn(model_outputs[0], y)
    
    def parameters(self):
        return self.model.parameters()
    
    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def move_to_device(self, device):
        self.model = self.model.to(device)

def inception(num_classes, pretrained, aux_logits):
    return InceptionNetV3(num_classes, pretrained, aux_logits)
