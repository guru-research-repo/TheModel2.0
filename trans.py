"""
transforms.py

Provides five image transformation functions:

- four_random_crops: generates four random square-like crops from a single image; does not support batch input.
- rotate: rotates images by a random angle within ±max_deg or 180° if inverse=True; supports both single-image (C,H,W) and batch (N,C,H,W) input, applying the same angle across the batch.
- foveation: simulates foveal blur via multi-scale Gaussian pyramids centered on a point in a tensor image; single-image only.
- logpolar_cv2: applies OpenCV's log-polar mapping to a single image (tensor or PIL); no batch support.
- logpolar_manual: manually computes a log-polar transform in PyTorch with optional smoothing and custom center selection; supports batch input.

Each function returns an output with the same shape as its input. Batch support is indicated per-function above.

"""

import random
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

class RandomCrop(torch.nn.Module):
    """
    Given an image tensor, return a list of 4 random square-ish crops.

    Args:
        n (int): Number of crops
        crop_size: Output image shape.
        data: input data tensor of size (B, C, W, H)

    Returns:
        torch.Tensor: Randomly cropped patches of shape (N, C, W, H)
    """
    def __init__(self, n: int = 4, crop_size: int = 180):
        super().__init__()
        self.n=n
        self.crop=T.RandomCrop(crop_size)

    def __call__(self, data):
        out = torch.stack([self.crop(data) for _ in range(self.n)])
        out = out.flatten(0,1) # output shape is (B*N,...), represented as B B B B
        # out = [self.crop(data)]
        # img = out[0]
        # out_img = TF.to_pil_image(img.clamp(0, 1))
        # filename = f"out/img_proc00.png"
        # out_img.save(filename)
        return out
    
class Rotate(torch.nn.Module):
    """
    Rotate the image tensor by a random angle in [-max_deg, max_deg],
    or by 180° if inverse=True.

    Args:
        data (torch.Tensor): Input image of shape (C, H, W) or batch (N, C, H, W).
        max_deg (float): Maximum absolute rotation angle (±max_deg).
        inverse (bool): If True, rotate by exactly 180° instead of a random angle.

    Returns:
        torch.Tensor: Rotated image(s), same shape as input.
    """
    def __init__(self, deg: float = 15.0, invert: bool = False):
        super().__init__()
        if invert:
            self.rotate = T.RandomRotation((180,180))
        else:
            self.rotate = T.RandomRotation(deg)

    def __call__(self, data):
        # print(data.shape)
        out = self.rotate(data)
        # img = out[0]
        # out_img = TF.to_pil_image(img.clamp(0, 1))
        # filename = f"out/img_proc01.png"
        # out_img.save(filename)
        return out


class Foveate(torch.nn.Module):
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
        shape = img.shape[:2]
        data = img.flatten(0,1)
        out = self.foveat_img(data, [(self.crop_size / 2, self.crop_size / 2)]).unflatten(dim=0, sizes=shape).float()
        # img = out[0]
        # out_img = TF.to_pil_image(img.clamp(0, 1))
        # filename = f"out/img_proc02.png"
        # out_img.save(filename)
        return out

    def pyramid(self, tensor, sigma=1, prNum=6):
        C,H,W = tensor.shape[-3:]
        G = tensor.clone().unsqueeze(0)
        # print(G.shape)
        pyramids = [G]
        
        # gaussian blur
        blur = T.GaussianBlur(5, sigma)

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
        # fix shape back to original
        for i in range(1, prNum):
            pyramids[i] = F.interpolate(pyramids[i], size=(H,W))
        # self.timer.end('fix shape')

        # stack and remove the extra batch dim
        out = torch.stack(pyramids).squeeze()
        # print("pyramid out: ", out.shape)
        return out
    
    def foveat_img(self, im, fixs):
        """
        im: input image
        fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
        
        This function outputs the foveated image with given input image and fixations.
        """
        sigma = 0.248
        prNum = 6
        As = self.pyramid(im, sigma, prNum)  # shape: (prNum, C, H, W)
        H, W = im.shape[-2:]
        # parameters
        p = 7.5
        k = 3
        alpha = 2.5
        # grid
        x = torch.arange(W, device=As.device).float()
        y = torch.arange(H, device=As.device).float()
        x2d, y2d = torch.meshgrid(x, y, indexing='ij')
        # distance map
        theta = torch.sqrt((x2d - fixs[0][0])**2 + (y2d - fixs[0][1])**2) / p
        for fx, fy in fixs[1:]:
            theta = torch.minimum(theta, torch.sqrt((x2d - fx)**2 + (y2d - fy)**2) / p)
        R = alpha / (theta + alpha)
        # blending coefficients
        Ts = [torch.exp(-((2**(i-3) * R / sigma)**2) * k) for i in range(1, prNum)]
        Ts.append(torch.zeros_like(theta))
        # omega thresholds
        omega = np.zeros(prNum)
        for i in range(1, prNum):
            omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma
        omega = np.clip(omega, None, 1)
        # layer indices
        layer_ind = torch.zeros_like(R, dtype=torch.long)
        for i in range(1, prNum):
            mask = (R >= omega[i]) & (R <= omega[i-1])
            layer_ind[mask] = i
        # blend factors
        Bs = [(0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5) for i in range(1, prNum)]
        # masks
        Ms = torch.zeros((prNum, H, W), device=As.device)
        for i in range(prNum):
            mask_i = layer_ind == i
            if i == 0:
                Ms[i][mask_i] = 1
            else:
                Ms[i][mask_i] = 1 - Bs[i-1][mask_i]
            mask_i1 = layer_ind - 1 == i
            if mask_i1.any() and i < prNum:
                Ms[i][mask_i1] = Bs[i][mask_i1]
        # combine
        # print(Ms.unsqueeze(1).shape)
        # print(As.shape)
        im_fov = (Ms.unsqueeze(1) * As).sum(dim=0)
        return im_fov
    
class LogPolar(torch.nn.Module):
    def __init__(self, input_shape=None, output_shape=None, smoothing = 0, 
                 mask = False, position='circumscribed', log_polar_distance = 2, random_center = False,
                 device = 'cpu'):
        super().__init__()
        self.input_shape = input_shape
        self.default_center = input_shape[0] / 2, input_shape[1] / 2

        self.output_shape = output_shape
        self.smoothing = smoothing
        self.mask = mask
        self.position = position

        self.log_polar_distance = log_polar_distance
        self.random_center = random_center

        X, Y = self.compute_map(self.input_shape, self.output_shape)
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

        self.device = device

    def getPoints(self, numPoints, prob_arr, threshold = 0.20):
#         print("shape",prob_arr.shape)
        crop_size = 0
        prob_reshape = prob_arr.reshape(-1)
        
        y_threshold_amt = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        x_threshold_amt = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        border_mask = np.zeros_like(prob_arr)
        border_mask[y_threshold_amt:-y_threshold_amt, x_threshold_amt:-x_threshold_amt] = 1
        
#         print(y_threshold_amt, "x", x_threshold_amt)
#         print(border_mask[border_mask==1].shape)
        border_mask = border_mask.reshape(-1)

        prob_border_masked = prob_reshape * border_mask
        prob_border_masked /= prob_border_masked.sum()
        
#         print(prob_border_masked.shape, prob_border_masked.sum())

        try:
            points = np.random.choice(prob_reshape.shape[0], numPoints, p = prob_border_masked)
#             print("points", numPoints)
#             print("points", points)
            unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
            return unraveled_points
        except:
            print("errors")
#             return np.array()
#            print( prob_arr,prob_border_masked.sum())
            return np.random.choice(prob_reshape.shape[0], numPoints)
                
        
    def SaliencePoints(self, data):
    #         print("bb", data.shape, "aa", np.array(data).shape)
        
            cv2_img = cv2.cvtColor(np.array(data).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            
    #         print("cc", cv2_img.shape)
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = saliency.computeSaliency(cv2_img)
    #        my_map = FasaSaliencyMapping(cv2_img.shape[0], cv2_img.shape[1])  # init the saliency object
    #        saliencyMap = my_map.returnMask(cv2_img, tot_bins=8, format='BGR')/255.0
    #         print("dd", saliencyMap)
            points = self.getPoints(1, saliencyMap)
    #        print(points[0], points[1])
            return (points[0][0], points[1][0])
    
    def compute_map(self, input_shape, output_shape):
        input_shape_x, input_shape_y = input_shape
        
        if self.position == 'circumscribed':
            MAX_R = torch.log(torch.tensor(input_shape).float().norm() / 2 * self.log_polar_distance)
        else:
            MAX_R = torch.log(torch.tensor(input_shape).float().max() / 2 * self.log_polar_distance)

        theta, r = torch.meshgrid(torch.arange(self.output_shape[0]), torch.arange(self.output_shape[1]), indexing='ij')
        theta = theta.float()
        r = r.float()
        X = (torch.exp(r * MAX_R / self.output_shape[1])) * torch.cos(theta * 2 * torch.pi / self.output_shape[0])
        Y = (torch.exp(r * MAX_R / self.output_shape[1])) * torch.sin(theta * 2 * torch.pi / self.output_shape[0])

        mask = (0 <= X) & (X < input_shape_x) & (0 <= Y) & (Y < input_shape_y)

        return X, Y

    def compute_mask(self, X, Y, input_shape):
        return (0 <= X) & (X < input_shape[0]) & (0 <= Y) & (Y < input_shape[1])
    
    def forward(self, data, center_x = None, center_y = None):
        
        img = data.permute(0, 2, 3, 1)
        img = (img - img.min()) / (img.max() - img.min())
        
        if data.shape[-2:] != self.input_shape:
            X, Y = self.compute_map(data.shape[-2:], self.output_shape)
        else:
            X = self.get_buffer('X')
            Y = self.get_buffer('Y')
        # print("xy", X,Y)
        # print("input shape", self.input_shape)
        # print("output shape", self.output_shape)

        if not center_x or not center_y:
            center_y, center_x = self.default_center
            
        if self.random_center and random.random() > 0.4 :
            center_y, center_x = self.SaliencePoints(data)
            
        X = center_x + X
        Y = center_y - Y

        # print("centre", center_x, center_y )
        mask = (self.compute_mask(X, Y, self.input_shape)  if self.mask else torch.ones_like(X)).to(self.device)
        # print("mask", mask)
        if self.smoothing == None:
            return (
                mask * (
                    data[
                      ...,
                      Y.long().clamp(0, data.shape[-2] - 1),
                      X.long().clamp(0, data.shape[-1] - 1),
                      # Y.long() % (data.shape[-2] - 1),
                      # X.long() % (data.shape[-1] - 1)
                    ]
                )
            )
                
        y_down, x_down = Y.long().clamp(0, data.shape[-2] - 1), X.long().clamp(0, data.shape[-1] - 1)
        y_up, x_up = (y_down+1).clamp(0, data.shape[-2] - 1), (x_down+1).clamp(0, data.shape[-1] - 1)
        
        down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
        down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
        up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
        up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing

        total_dist = down_down_dist + down_up_dist +  up_down_dist +  up_up_dist
        
        down_down_weight = (down_down_dist / total_dist).to(self.device)
        down_up_weight = (down_up_dist / total_dist).to(self.device)
        up_down_weight = (up_down_dist / total_dist).to(self.device)
        up_up_weight = (up_up_dist / total_dist).to(self.device)

        return (
            mask * (
                down_down_weight * data[...,y_down,x_down] +
                down_up_weight * data[...,y_down,x_up] +
                up_down_weight * data[...,y_up,x_down] +
                up_up_weight * data[...,y_up,x_up]
            )
        )

class Pipeline(torch.nn.Module):
    def __init__(self, type = 'train', logpolar = False, device = 'cpu', 
                 n_crops = 4,
                 normalize = T.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)),
                crop_size = 180,
                output_shape = (180,180)):
        """
        Create transformation pipeline
        type = 'train', 'inverted', or None
        """
        super().__init__()

        if type == 'train':
            self.crop = RandomCrop(n=n_crops, crop_size=crop_size)
            self.rotate = Rotate()
        elif type == 'inverted':
            self.crop = RandomCrop(n=n_crops, crop_size=crop_size)
            self.rotate = Rotate(invert = True)
        else:
            self.crop = RandomCrop(n=n_crops, crop_size=crop_size)
            self.rotate = torch.nn.Identity()

        
        if logpolar:
            self.foveate = Foveate(crop_size=crop_size)
            self.logpolar = LogPolar(input_shape=(crop_size, crop_size),
                                 output_shape=output_shape, device=device)
        else:
            self.foveate = torch.nn.Identity()
            self.logpolar = torch.nn.Identity()
        
        self.normalize = normalize
        self.tensorize = T.ToTensor()
        
        self.compose = T.Compose([
            self.crop,
            self.rotate,
            self.foveate,
            self.logpolar,
            # self.normalize
        ])

    def forward(self, data):
        if not isinstance(data, torch.Tensor): 
            data = self.tensorize(data)
        # print(data.shape)
        out = self.compose(data)
        # img = out[0]
        # out_img = TF.to_pil_image(img.clamp(0, 1))
        # filename = f"out/img_proc03.png"
        # out_img.save(filename)
        # print(out.shape)
        return out
        # return self.compose(data)
