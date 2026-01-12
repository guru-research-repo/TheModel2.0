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

n_threads = 40

os.environ["OMP_NUM_THREADS"] = str(n_threads)       # or 32, or whatever you think is reasonable
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"]  = str(n_threads)

torch.set_num_threads(n_threads)
torch.set_num_interop_threads(2)

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

    img = Foveate().foveat_img(img, fixs=[(112, 112)])
    img = LogPolar(input_shape=(224, 224), output_shape=(224, 224)) \
                .forward(img.unsqueeze(0), center_x = 112, center_y = 112)
    
    with torch.no_grad():
        output = model(img)
        # pred_class = torch.argmax(output, dim=1).item()   # Python int
        # pred_logit = torch.max(output, dim=1).values.item()
    
    return output.squeeze(0)[26].item()


if __name__ == "__main__":
    # img_path = f'data/faces/faces/32_identities/test/EmmanuelMacron/27.jpg' # image or directory of images
    img_path = f'data/faces/faces/128_identities/train/JackMa/10.jpg'
    img_path = Path(img_path).expanduser()
    image = Image.open(img_path).convert("RGB")
    tensor_img = TF.to_tensor(image)
    tensor_img = TF.rotate(tensor_img, angle=180)  # rotate to correct orientation if needed

    # -------------- Kernel Here ------------------
    heatmap = np.zeros((224, 224))
    for x in range(224):
        print(f"processing x = {x} now")
        for y in range(224):
            heatmap[x, y] = kernel(tensor_img, model, cx=y, cy=x) # different axes
    # ---------------------------------------------

    tensor_img = (tensor_img.permute(1,2,0).numpy() * 255).astype(np.uint8) 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(tensor_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap='coolwarm')
    axes[1].set_title("Heatmap")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis("off")

    axes[2].imshow(tensor_img)
    axes[2].imshow(heatmap, cmap='coolwarm', alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig('./output/unfamiliar_inverted.png', dpi=150)
    plt.close(fig)

