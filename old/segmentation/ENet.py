import torch
import torch.nn as nn
import torch.nn.functional as F

class ENet():
    def __init__(self, num_classes: int = 10):
        self.name = "ENet"
        self.num_classes = num_classes
        self.model = ENetImpl(num_classes)

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

def enet(num_classes):
    return ENet(num_classes)


class ENetImpl(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial = InitialBlock()

        # Encoder
        self.encoder = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            *[Bottleneck(64, 64) for _ in range(4)],

            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=16)
        )

        # Decoder
        self.decoder = nn.Sequential(
            Bottleneck(128, 64, upsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 16, upsample=True),
            Bottleneck(16, 16)
        )

        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.fullconv(x)
        return x


class InitialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_branch = nn.Conv2d(3, 13, kernel_size=3, stride=2, padding=1, bias=False)
        self.ext_branch = nn.MaxPool2d(2, stride=2)
        self.batch_norm = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Pad ext with reflection to match main's spatial size
        if main.size()[2:] != ext.size()[2:]:
            diffY = main.size(2) - ext.size(2)
            diffX = main.size(3) - ext.size(3)

            # Apply reflection padding to bottom/right if needed
            ext = F.pad(ext, [0, diffX, 0, diffY], mode='reflect')

        out = torch.cat((main, ext), dim=1)
        out = self.batch_norm(out)
        return self.prelu(out)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False, dilated=1, asymmetric=False, dropout_prob=0.1):
        super().__init__()
        internal = in_channels // 4
        stride = 2 if downsample else 1

        # Main branch
        if downsample:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif upsample:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.main = nn.Identity()

        # Residual branch
        if upsample:
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(in_channels, internal, 1, stride=1, bias=False),
                nn.BatchNorm2d(internal),
                nn.PReLU(),

                nn.ConvTranspose2d(internal, internal, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(internal),
                nn.PReLU(),

                nn.ConvTranspose2d(internal, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=dropout_prob)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, internal, 1, stride=1, bias=False),
                nn.BatchNorm2d(internal),
                nn.PReLU(),

                nn.Conv2d(internal, internal, 3, stride=stride, padding=dilated, dilation=dilated, bias=False),
                nn.BatchNorm2d(internal),
                nn.PReLU(),

                nn.Conv2d(internal, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=dropout_prob)
            )

        self.prelu = nn.PReLU()

    def forward(self, x):
        main = self.main(x)
        residual = self.residual(x)
        return self.prelu(main + residual)
