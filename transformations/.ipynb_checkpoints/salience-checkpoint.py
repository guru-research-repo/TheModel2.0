import numpy as np
import torch

class SalienceSampling(torch.nn.Module):

    def __init__(self, num_points, crop_size):
        self.num_points=num_points
        self.crop_size=crop_size

    def __call__(self, img, salience_map):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        def getPoints(numPoints, prob_arr, threshold = 0.15):
            prob_reshape = prob_arr.reshape(-1)

            y_threshold_amt = max(self.crop_size // 2, int(threshold * prob_arr.shape[0]))
            x_threshold_amt = max(self.crop_size // 2, int(threshold * prob_arr.shape[0]))
            border_mask = np.zeros_like(prob_arr)
            border_mask[y_threshold_amt:-y_threshold_amt, x_threshold_amt:-x_threshold_amt] = 1
            border_mask = border_mask.reshape(-1)

            prob_border_masked = prob_reshape * border_mask
            prob_border_masked /= prob_border_masked.sum()

            try:
                points = np.random.choice(prob_reshape.shape[0], numPoints, p = prob_border_masked)
                unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
                return unraveled_points
            except:
                print(prob_border_masked.sum())

        def croppedTensors(points, tensor):
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

        points = getPoints(self.num_points, salience_map.numpy())

        images = croppedTensors(points, img)

        return torch.stack(images)

    def getSalienceMap(salience_path, image_path):
        path = image_path.split('/')
        salience = np.load('{}/{}/{}/{}.npy'.format(salience_path, path[-3], path[-2], path[-1].split('.jpg')[0]))
        return salience
