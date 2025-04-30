import os
import sys

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PSPNet import PSPNet

modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)
from data.FloodNet import FloodNetSegmentation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

current_cuda_device = None
print(f"is cuda available: {torch.cuda.is_available()}")
print(f"cuda device count: {torch.cuda.device_count()}")
current_cuda_device = torch.cuda.current_device()
print(f"current cuda device: {current_cuda_device}")
print(f"current cuda device name: {torch.cuda.get_device_name(current_cuda_device)}")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
    #                      std=[0.229, 0.224, 0.225])
])

dataset = FloodNetSegmentation(img_dir="../data/FloodNet_dataset/train/image",
                               mask_dir="../data/FloodNet_dataset/train/label",
                               transform=transform
                               )

train_data = DataLoader(dataset, batch_size=64, shuffle=True)

model = PSPNet()
model.move_to_device(device)
model.train_mode()

import torch.optim as optim
import torch.nn as nn
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 1
for epoch in range(num_epochs):
    size = len(train_data.dataset)
    current = 0
    running_loss = 0.0
    print(f"-------------- Epoch {epoch+1} --------------")
    for batch, (images, masks) in enumerate(train_data):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, masks.squeeze(1))  # squeeze if mask is [B,1,H,W]
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), current + len(images)
        running_loss += loss
        print(f"batch: {batch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from PIL import Image

def infer(model, image_path, device):
    model.eval_mode()
    image = Image.open(image_path).convert("RGB")
    input_image = TF.resize(image, (299, 299))
    input_tensor = TF.to_tensor(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.forward(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return image, pred

def visualize_segmentation(original, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap='jet', alpha=0.7)
    plt.axis('off')
    plt.savefig("seg")

image_path = "../data/FloodNet_dataset/test/image/6336.jpg"
image, pred_mask = infer(model, image_path, device)
visualize_segmentation(image, pred_mask)
