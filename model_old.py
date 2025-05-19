import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False):
        super(Model, self).__init__()

        weights = ResNet18_Weights.IMAGENET1K_V2 if pretrained else None

        self.resnet_model = resnet18(weights=weights)

        # self.resnet_model = torchvision.models.resnet50(pretrained = False)
        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256 ,num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Replaces torch.squeeze(x)
        #x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """
        Returns the 512‐dim vector just before the two FC layers.
        """
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x