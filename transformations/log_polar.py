import torch
import torch.nn.functional as F
import torchvision

import cv2
import numpy as np
import random
# from deepgaze.saliency_map import FasaSaliencyMapping
import matplotlib.pyplot as plt

def getPoints(numPoints, prob_arr, threshold = 0.20):
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
                
    
def SaliencePoints(data):
#         print("bb", data.shape, "aa", np.array(data).shape)
       
        cv2_img = cv2.cvtColor(np.array(data).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
        
#         print("cc", cv2_img.shape)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(cv2_img)
#        my_map = FasaSaliencyMapping(cv2_img.shape[0], cv2_img.shape[1])  # init the saliency object
#        saliencyMap = my_map.returnMask(cv2_img, tot_bins=8, format='BGR')/255.0
#         print("dd", saliencyMap)
        points = getPoints(1, saliencyMap)
#        print(points[0], points[1])
        return (points[0][0], points[1][0])
    
class LogPolar(torch.nn.Module):
    def __init__(self, input_shape=None, output_shape=None, smoothing = 0, mask = True, position='circumscribed', log_polar_distance = 2, random_center = False):
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
    
    def forward(self, data, center_x = None, center_y = None):
        
        img = data.permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.axis('off')
        plt.show()
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
            center_y, center_x = SaliencePoints(data)
            
        X = center_x + X
        Y = center_y - Y

        # print("centre", center_x, center_y )
        mask = self.compute_mask(X, Y, self.input_shape)  if self.mask else torch.ones_like(X)
        print("mask", mask)
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
        
        
        blur = torchvision.transforms.GaussianBlur(5, 1)
        
        y_down, x_down = Y.long().clamp(0, data.shape[-2] - 1), X.long().clamp(0, data.shape[-1] - 1)
        y_up, x_up = (y_down+1).clamp(0, data.shape[-2] - 1), (x_down+1).clamp(0, data.shape[-1] - 1)
        
        down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
        down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
        up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
        up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing

        total_dist = down_down_dist + down_up_dist +  up_down_dist +  up_up_dist
        
        down_down_weight = down_down_dist / total_dist
        down_up_weight = down_up_dist / total_dist
        up_down_weight = up_down_dist / total_dist
        up_up_weight = up_up_dist / total_dist

        return (
            mask * (
            # (
                down_down_weight * data[...,y_down,x_down] +
                down_up_weight * data[...,y_down,x_up] +
                up_down_weight * data[...,y_up,x_down] +
                up_up_weight * data[...,y_up,x_up]
            )
        )
        
        # return (
        #     mask * (
        #         data[
        #           ...,
        #           Y.long().clamp(0, data.shape[-1] - 1),
        #           X.long().clamp(0, data.shape[-2] - 1)
        #         ]
        #     )
        # )
            
if __name__ == '__main__':
    # from skimage.transform import warp_polar
    data = torch.arange(3*180*180).reshape(3,180,180).float()
    output_shape = (190,165)

    import timeit
    N = 1000

    lp = LogPolar(output_shape)

    # reference = timeit.timeit(lambda: torch.from_numpy(warp_polar(data.numpy(),scaling='log', output_shape=output_shape, channel_axis=0)), number=N)
    # print('ref', reference)
    implemented = timeit.timeit(lambda: lp(data), number=N)
    print('implemented', implemented)

    data = data.cuda()
    print('CUDA')

    # reference = timeit.timeit(lambda: torch.from_numpy(warp_polar(data.cpu().numpy(),scaling='log', output_shape=output_shape, channel_axis=0)), number=N)
    # print('ref', reference)
    implemented = timeit.timeit(lambda: lp(data), number=N)
    print('log polar', implemented)

