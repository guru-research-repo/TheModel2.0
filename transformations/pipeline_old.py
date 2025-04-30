import torch
import torch.nn as nn
from torchvision import transforms
from transformations import (
        SalienceSamplingOld as SalienceSampling, 
        LogPolar, 
        NRandomCrop, 
        Compose, 
        Resize, 
        Foveation, 
        FoveationOld, 
        Identity
)

import time
from utils import Timer

class Pipeline(torch.nn.Module):
    def __init__(self, 
            salience_path, 
            crop_size, 
            max_rotation, 
            lp, 
            lp_out_shape, 
            augmentation, 
            points, 
            inversion, 
            foveat,
            jitter_strength = 1.0,
            normalize = transforms.Normalize( mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225) ), 
            count = 1,
            flatten = torch.nn.Identity()):
        
        super().__init__()

        self.normalize = normalize

        self.salience_path = salience_path

        if augmentation == 'salience':
            self.transform = SalienceSampling(points, crop_size)
        elif augmentation == 'random': 
            self.transform = NRandomCrop(points, crop_size)
        else:
            self.transform = Resize(points, crop_size)

        if foveat:
            self.foveat = Foveation(crop_size=crop_size)
        else:
            self.foveat = Identity()

        if max_rotation == 0:
            self.rotate = Identity()
        elif inversion:
            self.rotate = transforms.RandomRotation((180,180))
        else:
            self.rotate = transforms.RandomRotation(max_rotation)

        if lp:
            self.lp = LogPolar(input_shape = (crop_size, crop_size), output_shape = lp_out_shape)
        else:
            self.lp = Identity()

        self.color = Identity()
        # self.color = transforms.Compose([ 
        #     transforms.RandomApply([transforms.ColorJitter( 0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength, 0.2 * jitter_strength)], p=0.8), 
        #     transforms.RandomGrayscale(p=0.2)
        #     ])

        self.count = count

        self.flatten = flatten

        self.timer = Timer()

    def forward(self, image, path = None, salience_map = None):

        orig_image = image
        if type(self.transform) == SalienceSampling and salience_map is None:
            salience_map = SalienceSampling.getSalienceMap(self.salience_path, path)

        l = []
        for _ in range(self.count):
            t0 = time.time_ns()

            image = orig_image.clone()
            self.timer.log('clone', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.transform(image, salience_map)
            self.timer.log('transform', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.foveat(image)
            self.timer.log('foveat', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.color(image)
            self.timer.log('color', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.rotate(image)
            self.timer.log('rotate', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.lp(image)
            self.timer.log('logpolar', time.time_ns() - t0)
            t0 = time.time_ns()

            image = self.normalize(image)
            self.timer.log('normalize', time.time_ns() - t0)
            t0 = time.time_ns()

            l.append(image)

        t0 = time.time_ns()
        l = torch.stack(l)
        self.timer.log('stack', time.time_ns() - t0)
        t0 = time.time_ns()
        
        l = self.flatten(l)
        self.timer.log('flatten', time.time_ns() - t0)
        t0 = time.time_ns()

        return l
