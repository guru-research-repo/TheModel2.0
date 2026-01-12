from model import Model
from pathlib import Path
from PIL import Image
from trans import Foveate, LogPolar
import matplotlib.image as mpimg
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

model = Model(size=224)
state_dict = torch.load("output/resnet18_lp_20251112_103918.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

def crop(img, cx, cy, crop_w=224, crop_h=224):
    # img: tensor [..., H, W] (for PIL, use img.size instead of shape)
    h, w = img.shape[-2], img.shape[-1]

    # desired crop box: [cx, cx+crop_w) x [cy, cy+crop_h)
    left = int(cx - 112)
    top  = int(cy - 112)
    right  = left + crop_w
    bottom = top  + crop_h

    # padding needed
    pad_left   = max(0, -left)
    pad_top    = max(0, -top)
    pad_right  = max(0, right  - w)
    pad_bottom = max(0, bottom - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        # order = (left, top, right, bottom)
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        left += pad_left
        top  += pad_top

    return F.crop(img, top=top, left=left, height=crop_h, width=crop_w)

def kernel(img, model, cx, cy):
    # img: tensor [C, H, W]
    img = crop(img, cx=cx, cy=cy)

    # img = Foveate().foveat_img(img, fixs=[(112, 112)])
    # img = LogPolar(input_shape=(224, 224), output_shape=(224, 224)) \
    #             .forward(img.unsqueeze(0), center_x = 112, center_y = 112)
    # with torch.no_grad():
    #     output = model(img)
    
    return img


if __name__ == "__main__":
    img_path = f'data/faces/faces/4_identities/test/EmmanuelMacron/101.jpg' # image or directory of images
    # img_path = f'data/faces/faces/32_identities/train/EmmanuelMacron/5.jpg'
    img_path = Path(img_path).expanduser()
    image = Image.open(img_path).convert("RGB")
    tensor_img = TF.to_tensor(image)

    # -------------- Kernel Here ------------------
    heatmap = np.zeros((224, 224))
    x = 0
    y = 224
    tensor_img = kernel(tensor_img, model, cx=y, cy=x)
    tensor_img = tensor_img.squeeze(0)

    # ---------------------------------------------
    tensor_img = (tensor_img.permute(1,2,0).numpy() * 255).astype(np.uint8) 

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.imshow(tensor_img)

    plt.savefig('./output/experiment0.png', dpi=150)
    plt.close(fig)

