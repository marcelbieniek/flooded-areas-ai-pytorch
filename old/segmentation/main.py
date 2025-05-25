import os
import sys

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.models.PSPNet import PSPNet
from src.models.DeepLabV3 import DeepLabV3
from ENet import ENet
from src.models.ENet import ENetV2

import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

modules_path = os.path.abspath("..")
if modules_path not in sys.path:
    sys.path.append(modules_path)
from src.data.FloodNet import FloodNetSegmentation

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
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

def mask_transform(mask):
    mask = mask.resize((299, 299), resample=Image.BILINEAR)
    # mask = torch.from_numpy(np.array(mask)).long()  # shape [H, W], values 0-9
    mask = TF.pil_to_tensor(mask).long()
    return mask

train_dataset = FloodNetSegmentation(img_dir="../data/FloodNet_dataset/train/image",
                               mask_dir="../data/FloodNet_dataset/train/label",
                               image_transform=transform,
                               mask_transform=mask_transform
                               )

train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = ENetV2()
model.move_to_device(device)
model.train_mode()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 3
for epoch in range(num_epochs):
    size = len(train_data.dataset)
    current = 0
    running_loss = 0.0
    print(f"-------------- Epoch {epoch+1} --------------")
    for batch, (images, masks) in enumerate(train_data):
        images, masks = images.to(device), masks.to(device).squeeze().long()
        optimizer.zero_grad()
        outputs = model.forward(images)
        # print(outputs.shape)
        # print(masks.shape)
        # loss = criterion(outputs, masks.squeeze(1))  # squeeze if mask is [B,1,H,W]
        # if batch >= 336:
        #     print(masks)
        #     torch.save(masks, f"masks_{batch}.pt")
        # loss = criterion(outputs, masks)

        # _, _, h, w = outputs.shape
        # outputs = outputs[:, :, :masks.shape[1], :masks.shape[2]]

        loss = model.calculate_loss(criterion, outputs, masks)
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), current + len(images)
        running_loss += loss
        print(f"batch: {batch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

model.save_model("model.pth")





def infer(model, image_path, device):
    model.eval_mode()
    image = Image.open(image_path).convert("RGB")
    image = image.resize((299, 299), resample=Image.BILINEAR)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.forward(input_tensor)
        print(output.shape)
        print(output)
        # print(output[:, 0, 0])
        # output = torch.softmax(output, dim=0)
        # print(output[:, 0, 0])
        # print(output[:, 0, 0].sum())
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        # pred = torch.argmax(output, dim=0).byte().cpu().numpy()
        # np.savetxt("pred.txt", pred)
        print(pred.shape)
        print(pred)
        # print(pred.dtype)
    
    return image, pred

def visualize_segmentation(original, ground_truth, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(ground_truth)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig("seg")

def decode_segmap(mask, colormap):
    """Convert class indices to RGB image using a colormap."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colormap)):
        rgb[mask == i] = colormap[i]
    return rgb

# Define a colormap with 10 distinct colors
COLORMAP = [
    (0, 0, 0),        # class 0 - black
    (128, 0, 0),      # class 1 - maroon
    (0, 128, 0),      # class 2 - green
    (128, 128, 0),    # class 3 - olive
    (0, 0, 128),      # class 4 - navy
    (128, 0, 128),    # class 5 - purple
    (0, 128, 128),    # class 6 - teal
    (128, 128, 128),  # class 7 - gray
    (64, 0, 0),       # class 8 - brown
    (192, 192, 192),  # class 9 - light gray
]

model.load_model("model.pth")

image_path = "../data/FloodNet_dataset/test/image/6445.jpg"
ground_truth_path = "../data/FloodNet_dataset/test/label/6445_lab.png"

image, pred_mask = infer(model, image_path, device)

ground_truth_mask = Image.open(ground_truth_path).convert("L")
ground_truth_mask = mask_transform(ground_truth_mask).squeeze().numpy()
ground_truth = decode_segmap(ground_truth_mask, COLORMAP)

segmentation = decode_segmap(pred_mask, COLORMAP)

visualize_segmentation(image, ground_truth, segmentation)
