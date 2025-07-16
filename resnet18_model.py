import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False, use_resnet18=True):
        super(Model, self).__init__()

        if use_resnet18:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet18(weights=weights)
            self.feature_dim = 512
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet50(weights=weights)
            self.feature_dim = 2048

        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)

        self.fc1 = nn.Linear(self.feature_dim, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
