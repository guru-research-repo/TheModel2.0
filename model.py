import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False, hidden_neurons=100, num_iter=1, dropout=0.25):
        super(Model, self).__init__()

        weights = ResNet18_Weights.IMAGENET1K_V2 if pretrained else None

        self.resnet_model = resnet18(weights=weights)

        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256 ,num_classes)
        self.attractor = ConvolutionAttractor(hidden_neurons=hidden_neurons, N=num_iter, dropout=dropout)
        self.conv = nn.Conv2d(512, 512, kernel_size=3)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = self.attractor(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Replaces torch.squeeze(x)
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

class ConvolutionAttractor(nn.Module):
    # Input Shape: (batch_size, 512, 3, 3)
    # Output Shape: (batch_size, 512, 3, 3)
    def __init__(self, N=1, hidden_neurons=100, dropout=0.25):
        super(ConvolutionAttractor, self).__init__()
        
        self.skip = nn.Conv2d(512, 512, kernel_size=1)
        self.down = nn.Conv2d(512, 512, kernel_size=3)
        self.up   = nn.Upsample(size=(3, 3))
        self.activation = nn.Tanh()
        self.N = N
        self.dropout = nn.Dropout(dropout)
    
    def flow(self, x):
        x_skip = self.skip(x)
        x_hidden = self.up(self.down(x))
        x = self.activation(x_skip + x_hidden)
        return x
    
    def forward(self, x):
        x = self.dropout(x)
        for _ in range(self.N):
            x = self.flow(x)
        return x

class Attractor(nn.Module):
    # Input Shape: 512
    # Output Shape: 512
    def __init__(self, N=1, hidden_neurons=100, dropout=0.25):
        super(Attractor, self).__init__()
        
        self.skip = nn.Linear(512, 512)
        self.down = nn.Linear(512, hidden_neurons)
        self.up   = nn.Linear(hidden_neurons, 512)
        self.activation = nn.Tanh()
        self.N = N
        self.dropout = nn.Dropout(dropout)
    
    def flow(self, x):
        x_skip = self.skip(x)
        x_hidden = self.up(self.down(x))
        x = self.activation(x_skip + x_hidden)
        return x
    
    def forward(self, x):
        x = self.dropout(x)
        for _ in range(self.N):
            x = self.flow(x)
        return x