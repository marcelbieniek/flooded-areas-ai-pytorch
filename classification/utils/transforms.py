import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF

classification_image_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
    #                      std=[0.229, 0.224, 0.225])
])

segmentation_image_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

def segmentation_mask_transform(mask):
    mask = mask.resize((299, 299), resample=Image.BILINEAR)
    # mask = torch.from_numpy(np.array(mask)).long()  # shape [H, W], values 0-9
    mask = TF.pil_to_tensor(mask).long()
    return mask
