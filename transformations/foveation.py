import numpy as np
from PIL import Image
import torch
import numpy as np
import sys
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from utils import Timer

class Foveation(torch.nn.Module):

    def __init__(self, crop_size=None, p_val=None):
        super().__init__()
        self.crop_size=crop_size
        # self.timer = Timer()

    def __call__(self, img):
        """
        Args:
            img: tensor to be foveated.
        Returns:
            tensor: Foveated image.
        """
        return self.foveat_img(img, [(self.crop_size / 2, self.crop_size / 2)]).float()
        
        for i in range(img.shape[0]):
            img[i,:,:,:] = self.foveat_img(img[i].unsqueeze(0), [(self.crop_size / 2, self.crop_size / 2)]).squeeze(0)

        print('hello')
        imag = img.permute(1, 2, 0)
        imag = (imag - imag.min()) / (imag.max() - imag.min())
        plt.imshow(imag)
        plt.axis('off')
        plt.show()
        return img.float()

    def pyramid(self, tensor, sigma=1, prNum=6):
        C,H,W = tensor.shape[-3:]
        G = tensor.clone().unsqueeze(0)
        pyramids = [G]
        
        # gaussian blur
        blur = torchvision.transforms.GaussianBlur(5, sigma)

        # self.timer.start()
        # downsample
        for i in range(1, prNum):
            G = F.interpolate(blur(G), scale_factor = (0.5, 0.5), recompute_scale_factor=True)
            pyramids.append(G)
        # self.timer.end('downsample')
        
        # self.timer.start()
        # upsample
        for i in range(1, prNum):
            for _ in range(i):
                pyramids[i] = F.interpolate(pyramids[i], scale_factor = (2,2), mode='bilinear', align_corners=True, recompute_scale_factor=False)
        # self.timer.end('upsample')
        
        # self.timer.start()
        for i in range(1, prNum):
            pyramids[i] = F.interpolate(pyramids[i], size=(H,W))
        # self.timer.end('fix shape')

        return torch.stack(pyramids).squeeze()

    #def foveat_img(p, im, fixs):
    def foveat_img(self, im, fixs):
        """
        im: input image
        fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
        
        This function outputs the foveated image with given input image and fixations.
        """
        sigma=0.248
        prNum = 6
        As = self.pyramid(im, sigma, prNum)
        height, width = im.shape[-2:]
        
        # compute coef
        p = 7.5
        k = 3
        alpha = 2.5

        x = torch.arange(0, width, 1, device=As.device).float()
        y = torch.arange(0, height, 1, device=As.device).float()
        x2d, y2d = torch.meshgrid(x, y, indexing='ij')
        theta = torch.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
        for fix in fixs[1:]:
            theta = torch.minimum(theta, torch.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
        R = alpha / (theta + alpha)
        
        Ts = []
        for i in range(1, prNum):
            Ts.append(torch.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
        Ts.append(torch.zeros_like(theta))
        
        # Ts = torch.cat([torch.exp(-((2 ** (torch.arange(1,prNum)-3).reshape(-1,1,1)) * R / sigma) ** 2 * k), torch.zeros_like(theta).unsqueeze(0)])

        # omega
        omega = np.zeros(prNum)
        for i in range(1, prNum):
            omega[i-1] = torch.sqrt(torch.log(torch.tensor(2))/k) / (2**(i-3)) * sigma

        omega[omega>1] = 1
        # omega = torch.clamp(torch.sqrt(torch.log(torch.tensor(2))/k) / (2**(torch.arange(1, prNum)-3)) * sigma, max=1)

        # layer index
        layer_ind = torch.zeros_like(R)
        for i in range(1, prNum):
            ind = torch.logical_and(R >= omega[i], R <= omega[i - 1])
            layer_ind[ind] = i

        # B
        Bs = []
        for i in range(1, prNum):
            Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

        # M
        Ms = torch.zeros((prNum, R.shape[0], R.shape[1])).to(As.device)

        for i in range(prNum):
            ind = layer_ind == i
            if torch.sum(ind) > 0:
                if i == 0:
                    Ms[i][ind] = 1
                else:
                    Ms[i][ind] = 1 - Bs[i-1][ind]

            ind = layer_ind - 1 == i
            if torch.sum(ind) > 0:
                Ms[i][ind] = Bs[i][ind]

        im_fov = (Ms.unsqueeze(1) * As).sum(axis=0)

        return im_fov
