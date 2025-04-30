import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SalienceCrop(torch.nn.Module):
    def __init__(self, num_points, crop_size):
        super().__init__()
        self.num_points=num_points
        self.crop_size=crop_size

    def getPoints(self, numPoints, prob_arr, threshold = 0.15):
#         print(prob_arr.shape)
        prob_reshape = prob_arr.reshape(-1)
        
        y_threshold_amt = max(self.crop_size // 2, int(threshold * prob_arr.shape[0]))
        x_threshold_amt = max(self.crop_size // 2, int(threshold * prob_arr.shape[0]))
        border_mask = np.zeros_like(prob_arr)
        border_mask[y_threshold_amt:-y_threshold_amt, x_threshold_amt:-x_threshold_amt] = 1
        
#         print(y_threshold_amt, "x", x_threshold_amt)
#         print(prob_arr.shape, border_mask[border_mask==1].shape)
        border_mask = border_mask.reshape(-1)

        prob_border_masked = prob_reshape * border_mask
        prob_border_masked /= prob_border_masked.sum()
        
#         print(prob_border_masked.shape, prob_border_masked.sum())

        try:
            points = np.random.choice(prob_reshape.shape[0], numPoints, p = prob_border_masked)
#             print(numPoints)
#             print(points)
            unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
            return unraveled_points
        except:
            print(prob_border_masked.sum())
                
    def croppedTensors(self, points, tensor):
        crops = []
        n = points.shape[1]
        y1, x1 = points[0], points[1]

        left = (x1 - self.crop_size / 2).astype(int)
        top = (y1 - self.crop_size / 2).astype(int)
        right = (x1 + self.crop_size / 2).astype(int)
        bottom = (y1 + self.crop_size / 2).astype(int)

        for v in range(n):
            cropped_img = tensor[...,top[v]:bottom[v],left[v]:right[v]]

            crops.append(cropped_img)

        return crops
    
    def __call__(self, data):
#         print("bb", data.shape, "aa", np.array(data).shape)
       
        cv2_img = cv2.cvtColor(np.array(data).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
#         print("cc", cv2_img.shape)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(cv2_img)
#         print("dd", saliencyMap.shape)
        points = self.getPoints(self.num_points, saliencyMap)

        images = self.croppedTensors(points, data)

        for t in images:
            img = t.permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        
        return images[0]
    