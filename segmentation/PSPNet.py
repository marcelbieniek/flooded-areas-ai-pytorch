import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class PSPNet():
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        self.name = "PSPNet"
        self.num_classes = num_classes
        self.model = PSPNetImpl(num_classes, pretrained)
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=pretrained)
        # self.model.classifier = DeepLabHead(2048, num_classes)

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

def pspnet(num_classes, pretrained):
    return PSPNet(num_classes, pretrained)


class PSPNetImpl(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        resnet = resnet101(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # Final ResNet block

        self.ppm = PyramidPoolingModule(in_channels=2048)
        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // len(pool_sizes) * len(pool_sizes), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
