import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.transforms import functional as TF

classification_image_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    # transforms.RandomHorizontalFlip(0.5), # data augmentation
    # transforms.RandomVerticalFlip(0.5), # data augmentation
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # data augmentation
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
    #                      std=[0.229, 0.224, 0.225])
])

segmentation_image_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

def segmentation_mask_tf(mask):
    mask = mask.resize((299, 299), resample=Image.BILINEAR) # data augmentation
    # mask = torch.from_numpy(np.array(mask)).long()  # shape [H, W], values 0-9
    mask = TF.pil_to_tensor(mask).long()
    return mask

# data augmentation
seg_joint_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    # transforms.RandomResizedCrop((320, 320)),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
])

# 320
