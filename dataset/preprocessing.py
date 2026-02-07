import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


TRANSFORM_SIZE = 768
PATCH_SIZE = 48


transform = transforms.Compose([
    transforms.Resize((TRANSFORM_SIZE, TRANSFORM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])




def process_image_and_coords(image, coords):
    image_pil = Image.fromarray(image)
    orig_w, orig_h = image_pil.size


    scale_x = TRANSFORM_SIZE / orig_w
    scale_y = TRANSFORM_SIZE / orig_h


    coords_rescaled = coords.copy()
    for i in range(coords.shape[0]):
        for j in range(4):
            coords_rescaled[i][j] = int(coords_rescaled[i][j] * scale_x)
        for j in range(4, 8):
            coords_rescaled[i][j] = int(coords_rescaled[i][j] * scale_y)


    image_tensor = transform(image_pil)
    return image_tensor, torch.tensor(coords_rescaled)




def image_to_patches_reshape(img, patch_size=PATCH_SIZE):
    C, H, W = img.shape
    img = img.view(C, H // patch_size, patch_size, W // patch_size, patch_size)
    img = img.permute(1, 3, 0, 2, 4)
    return img.reshape(-1, C, patch_size, patch_size)