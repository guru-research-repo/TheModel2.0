from skimage.transform import warp_polar
import torch
class LogPolar(object):

    def __init__(self, center=None,radius=None, output_shape=None , scaling = 'log'):
        self.center=center
        self.radius=radius
        self.output_shape=output_shape
        self.scaling=scaling

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """

        return torch.from_numpy(warp_polar(img.numpy(),center= self.center,radius=self.radius,output_shape=self.output_shape,scaling=self.scaling, channel_axis=0))
