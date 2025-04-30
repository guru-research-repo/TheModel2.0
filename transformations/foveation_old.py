import numpy as np
from PIL import Image
import torch
import numpy as np
import sys
import cv2

class Foveation(torch.nn.Module):

    def __init__(self, crop_size=None, p_val=None):
        self.crop_size=crop_size

    def __call__(self, img):
        """
        Args:
            img: tensor to be foveated.
        Returns:
            tensor: Foveated image.
        """

        img = torch.flatten(img, 0, 1).transpose(0,2)
        
        img = torch.from_numpy(self.foveat_img(img.numpy(), [(self.crop_size / 2, self.crop_size / 2)]))
        return img.transpose(0,2).unsqueeze(0).float()

    def genGaussiankernel(self, width, sigma):
        x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
        x2d, y2d = np.meshgrid(x, x)
        kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / np.sum(kernel_2d)
        return kernel_2d

    def pyramid(self, im, sigma=1, prNum=6):
        height_ori, width_ori, ch = im.shape
        G = im.copy()
        pyramids = [G]
        
        # gaussian blur
        Gaus_kernel2D = self.genGaussiankernel(5, sigma)
        
        # downsample
        for i in range(1, prNum):
            G = cv2.filter2D(G, -1, Gaus_kernel2D)
            height, width, _ = G.shape
            G = cv2.resize(G, (int(width/2), int(height/2)))
            pyramids.append(G)
        
        
        # upsample
        for i in range(1, 6):
            curr_im = pyramids[i]
            for j in range(i):
                if j < i-1:
                    im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
                else:
                    im_size = (width_ori, height_ori)
                curr_im = cv2.resize(curr_im, im_size)
                curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
            pyramids[i] = curr_im

        return pyramids

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
        height, width, _ = im.shape
        
        # compute coef
        p = 7.5
        k = 3
        alpha = 2.5

        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)
        x2d, y2d = np.meshgrid(x, y)
        theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
        for fix in fixs[1:]:
            theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
        R = alpha / (theta + alpha)
        
        Ts = []
        for i in range(1, prNum):
            Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
        Ts.append(np.zeros_like(theta))

        # omega
        omega = np.zeros(prNum)
        for i in range(1, prNum):
            omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

        omega[omega>1] = 1

        # layer index
        layer_ind = np.zeros_like(R)
        for i in range(1, prNum):
            ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
            layer_ind[ind] = i

        # B
        Bs = []
        for i in range(1, prNum):
            Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

        # M
        Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

        for i in range(prNum):
            ind = layer_ind == i
            if np.sum(ind) > 0:
                if i == 0:
                    Ms[i][ind] = 1
                else:
                    Ms[i][ind] = 1 - Bs[i-1][ind]

            ind = layer_ind - 1 == i
            if np.sum(ind) > 0:
                Ms[i][ind] = Bs[i][ind]

        im_fov = (np.expand_dims(np.stack(Ms), -1) * np.stack(As)).sum(axis=0)

        # generate periphery image
        im_fov = np.zeros_like(As[0], dtype=np.float32)
        for M, A in zip(Ms, As):
            for i in range(3):
                im_fov[:, :, i] += np.multiply(M, A[:, :, i])

        im_fov = im_fov.astype(np.uint8)
        return im_fov
