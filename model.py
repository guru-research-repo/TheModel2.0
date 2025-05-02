import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False):
        super(Model, self).__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        self.resnet_model = resnet50(weights=weights)

        # self.resnet_model = torchvision.models.resnet50(pretrained = False)
        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Replaces torch.squeeze(x)
        #x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x