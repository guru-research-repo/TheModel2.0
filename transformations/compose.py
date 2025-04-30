import torch
class Compose(torch.nn.Module):
    def __init__(self, augmentations = []):
        super().__init__()
        self.augmentations = augmentations

    def forward(self, *args):
        for augment in self.augmentations:
            args = augment(*args) 
            if type(args) != tuple:
                args = (args, )
        return args[0]
