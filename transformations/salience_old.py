import numpy as np
import torch

class SalienceSampling(object):

    def __init__(self, num_points, crop_size, salience_path):
        self.num_points=num_points
        self.crop_size=crop_size
        self.salience_path = salience_path

    def __call__(self, img, path):
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

            points = np.random.choice(prob_reshape.shape[0], numPoints, p = prob_border_masked)
            unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
            return unraveled_points

        def croppedTensors(points, tensor):
            crops = []
            n = points.shape[1]
                
            for v in range(n):
                y1,x1 = points[:,v]

                # large crop image
                left = int(x1 - self.crop_size / 2)
                top = int(y1 - self.crop_size / 2)
                right = int(x1 + self.crop_size / 2)
                bottom = int(y1 + self.crop_size / 2)
                cropped_img = tensor[:,top:bottom,left:right]

                crops.append(cropped_img)

            return crops

        salience = SalienceSampling.getSalienceMap(self.salience_path, path)

        points = getPoints(self.num_points, salience)

        images = croppedTensors(points, img)

        return torch.stack(images)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def getSalienceMap(salience_path, image_path):
        path = image_path.split('/')
        salience = np.load('{}/{}/{}/{}.npy'.format(salience_path, path[-3], path[-2], path[-1].split('.jpg')[0]))
        return salience

