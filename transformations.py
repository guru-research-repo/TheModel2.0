import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
import cv2
import numpy as np

# Take four random crops of each image
class FourRandomCrops:
    def __init__(self, crop_scale=0.65):
        self.crop_scale = crop_scale

    def __call__(self, img):
        w, h = img.size
        crop_w = int(w * (self.crop_scale ** 0.5))  
        crop_h = int(h * (self.crop_scale ** 0.5))

        crops = []
        for _ in range(4):
            left = random.randint(0, w - crop_w) if w != crop_w else 0
            top = random.randint(0, h - crop_h) if h != crop_h else 0
            crop = TF.crop(img, top, left, crop_h, crop_w)
            crops.append(crop)
        return crops

# ------------------------------------------------------------------------
# Foveation and Log Polar Transform from original Model2.0 Code
# ------------------------------------------------------------------------

# Foveate image
class Foveation(torch.nn.Module):
    def __init__(self, crop_size, sigma=1, prNum=6):
        super().__init__()
        self.crop_size = crop_size
        self.sigma = sigma
        self.prNum = prNum

    def forward(self, img):
        if img.dim() != 3:
            raise ValueError(f"Expected image tensor to have 3 dimensions (C, H, W), got {img.dim()}")
        return self.foveat_img(img, [(self.crop_size / 2, self.crop_size / 2)]).float()

    def pyramid(self, tensor):
        C, H, W = tensor.shape
        G = tensor.unsqueeze(0)
        pyramids = [G]
        blur = T.GaussianBlur(5, sigma=self.sigma)

        for _ in range(1, self.prNum):
            G = F.interpolate(blur(G), scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            pyramids.append(G)

        for i in range(1, self.prNum):
            G_up = pyramids[i]
            for _ in range(i):
                G_up = F.interpolate(G_up, scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=False)
            pyramids[i] = F.interpolate(G_up, size=(H, W), mode='bilinear', align_corners=True)

        return torch.stack(pyramids).squeeze(1)

    def foveat_img(self, im, fixs):
        sigma = self.sigma
        prNum = self.prNum
        As = self.pyramid(im)
        C, H, W = im.shape

        p, k, alpha = 7.5, 3, 2.5
        x = torch.arange(0, W, device=im.device).float()
        y = torch.arange(0, H, device=im.device).float()
        x2d, y2d = torch.meshgrid(x, y, indexing='ij')
        theta = torch.sqrt((x2d - fixs[0][0])**2 + (y2d - fixs[0][1])**2) / p

        for fix in fixs[1:]:
            theta = torch.minimum(theta, torch.sqrt((x2d - fix[0])**2 + (y2d - fix[1])**2) / p)

        R = alpha / (theta + alpha)

        Ts = [torch.exp(-((2**(i-3)) * R / sigma)**2 * k) for i in range(1, prNum)]
        Ts.append(torch.zeros_like(theta))

        omega = torch.zeros(prNum, device=im.device)
        for i in range(1, prNum):
            omega[i-1] = torch.sqrt(torch.log(torch.tensor(2.0)) / k) / (2**(i-3)) * sigma
        omega = torch.clamp(omega, max=1.0)

        layer_ind = torch.zeros_like(R)
        for i in range(1, prNum):
            mask = (R >= omega[i]) & (R <= omega[i-1])
            layer_ind[mask] = i

        Bs = [(0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5) for i in range(1, prNum)]
        Ms = torch.zeros((prNum, H, W), device=im.device)

        for i in range(prNum):
            mask = layer_ind == i
            if i == 0:
                Ms[i][mask] = 1
            else:
                Ms[i][mask] = 1 - Bs[i-1][mask]

            mask_prev = (layer_ind - 1) == i
            if mask_prev.any():
                Ms[i][mask_prev] = Bs[i][mask_prev]

        return (Ms.unsqueeze(1) * As).sum(dim=0)
    
def getPoints(numPoints, prob_arr, threshold=0.20):
    prob_reshape = prob_arr.reshape(-1)
    y_threshold_amt = int(threshold * prob_arr.shape[0])
    x_threshold_amt = int(threshold * prob_arr.shape[1])
    border_mask = np.zeros_like(prob_arr)
    border_mask[y_threshold_amt:-y_threshold_amt, x_threshold_amt:-x_threshold_amt] = 1
    border_mask = border_mask.reshape(-1)

    prob_border_masked = prob_reshape * border_mask
    prob_border_masked_sum = prob_border_masked.sum()

    if prob_border_masked_sum == 0:
        prob_border_masked = np.ones_like(prob_border_masked) / prob_border_masked.size
    else:
        prob_border_masked /= prob_border_masked_sum

    points = np.random.choice(prob_reshape.shape[0], numPoints, p=prob_border_masked)
    unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
    return unraveled_points

def SaliencePoints(data):
    cv2_img = cv2.cvtColor(np.array(data).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(cv2_img)
    points = getPoints(1, saliencyMap)
    return (points[0][0], points[1][0])

# Log Polar Transformation
class LogPolar(torch.nn.Module):
    def __init__(self, input_shape=None, output_shape=None, smoothing = 0, mask = True, position='circumscribed', log_polar_distance = 700, random_center = False):
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

    def forward(self, data, center_x=None, center_y=None):
        H, W = data.shape[-2:]  # Get current input dimensions
    
        # Always recompute map for current size
        X, Y = self.compute_map((H, W), self.output_shape)
    
        # Decide center
        if center_x is None or center_y is None:
            if self.random_center and random.random() > 0.4:
                center_y, center_x = SaliencePoints(data)
                # Clamp to valid range
                center_y = np.clip(center_y, 0, H-1)
                center_x = np.clip(center_x, 0, W-1)
            else:
                center_y, center_x = H / 2, W / 2  # Middle of the current image
    
        # Shift map
        X = center_x + X
        Y = center_y - Y
    
        # Compute valid mask
        mask = self.compute_mask(X, Y, (H, W)) if self.mask else torch.ones_like(X)
    
        # No smoothing
        if self.smoothing is None:
            return (
                mask * (
                    data[
                        ...,
                        Y.long().clamp(0, H-1),
                        X.long().clamp(0, W-1)
                    ]
                )
            )
    
        # With smoothing
        blur = torchvision.transforms.GaussianBlur(5, 1)
    
        y_down, x_down = Y.long().clamp(0, H-1), X.long().clamp(0, W-1)
        y_up, x_up = (y_down+1).clamp(0, H-1), (x_down+1).clamp(0, W-1)
    
        down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
        down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
        up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
        up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing
    
        total_dist = down_down_dist + down_up_dist + up_down_dist + up_up_dist
    
        down_down_weight = down_down_dist / total_dist
        down_up_weight = down_up_dist / total_dist
        up_down_weight = up_down_dist / total_dist
        up_up_weight = up_up_dist / total_dist
    
        return (
            mask * (
                down_down_weight * data[..., y_down, x_down] +
                down_up_weight * data[..., y_down, x_up] +
                up_down_weight * data[..., y_up, x_down] +
                up_up_weight * data[..., y_up, x_up]
            )
        )

    
#     def forward(self, data, center_x = None, center_y = None):
#         if data.shape[-2:] != self.input_shape:
#             X, Y = self.compute_map(data.shape[-2:], self.output_shape)
#         else:
#             X = self.get_buffer('X')
#             Y = self.get_buffer('Y')

#         if not center_x or not center_y:
#             center_y, center_x = self.default_center
            
#         if self.random_center and random.random() > 0.4 :
#             center_y, center_x = SaliencePoints(data)
            
#         X = center_x + X
#         Y = center_y - Y

# #         print("centre", center_x, center_y )
#         mask = self.compute_mask(X, Y, self.input_shape)  if self.mask else torch.ones_like(X)
#         if self.smoothing == None:
#             return (
#                 mask * (
#                     data[
#                       ...,
#                       Y.long().clamp(0, data.shape[-2] - 1),
#                       X.long().clamp(0, data.shape[-1] - 1),
#                       # Y.long() % (data.shape[-2] - 1),
#                       # X.long() % (data.shape[-1] - 1)
#                     ]
#                 )
#             )
        
        
#         blur = T.GaussianBlur(5, 1)
        
#         y_down, x_down = Y.long().clamp(0, data.shape[-2] - 1), X.long().clamp(0, data.shape[-1] - 1)
#         y_up, x_up = (y_down+1).clamp(0, data.shape[-2] - 1), (x_down+1).clamp(0, data.shape[-1] - 1)
        
#         down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
#         down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
#         up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
#         up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing

#         total_dist = down_down_dist + down_up_dist +  up_down_dist +  up_up_dist
        
#         down_down_weight = down_down_dist / total_dist
#         down_up_weight = down_up_dist / total_dist
#         up_down_weight = up_down_dist / total_dist
#         up_up_weight = up_up_dist / total_dist

#         return (
#             mask * (
#                 down_down_weight * data[...,y_down,x_down] +
#                 down_up_weight * data[...,y_down,x_up] +
#                 up_down_weight * data[...,y_up,x_down] +
#                 up_up_weight * data[...,y_up,x_up]
#             )
#         )