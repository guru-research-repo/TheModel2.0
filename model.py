import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False, size = 180):
        super(Model, self).__init__()

        if size == 180:
            self.in_size = 512
        elif size == 224:
            self.in_size = 2048
        else:
            self.in_size = 512

        weights = ResNet18_Weights.IMAGENET1K_V2 if pretrained else None

        self.resnet_model = resnet18(weights=weights)

        # self.resnet_model = torchvision.models.resnet50(pretrained = False)
        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(self.in_size,256)
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.model(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Replaces torch.squeeze(x)
        #x = torch.squeeze(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)

        return x