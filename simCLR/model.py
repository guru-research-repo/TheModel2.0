# import torch
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights, resnet50


# class Model(torch.nn.Module):
#     def __init__(self, num_classes=32, pretrained=False, size = 180):
#         super(Model, self).__init__()

#         if size == 180:
#             self.in_size = 512
#         elif size == 224:
#             self.in_size = 2048
#         else:
#             self.in_size = 512

#         weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

#         self.resnet_model = resnet18(weights=weights)

#         # self.resnet_model = torchvision.models.resnet50(pretrained = False)
#         self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
#         self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
#         self.fc1 = nn.Linear(self.in_size,256)
#         self.fc2 = nn.Linear(256,num_classes)

#     def forward(self, x):
#         # print(x.shape)
#         x = self.model(x)
#         # print(x.shape)
#         x = self.avgpool(x)
#         # print(x.shape)
#         x = x.view(x.size(0), -1)  # Replaces torch.squeeze(x)
#         #x = torch.squeeze(x)
#         # print(x.shape)
#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class Model(nn.Module):
    def __init__(self, num_classes=32, pretrained=False, size=180, proj_dim=128):
        """
        ResNet18 backbone with two heads:
          • classifier head  → CE loss
          • projection head  → SupCon loss
        """
        super().__init__()

        # --------- Backbone feature dim (ResNet18 → 512) ---------
        self.in_size = 512

        # --------- Backbone (feature extractor) ---------
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(*(list(base.children())[:-2]))  # conv layers only
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------- Projection head (for SupCon) ---------
        self.proj_head = nn.Sequential(
            nn.Linear(self.in_size, self.in_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_size, proj_dim)
        )

        # --------- Classification head ---------
        self.fc1 = nn.Linear(self.in_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_feat=False):
        """
        Args:
            x : (B, 3, H, W)
            return_feat : if True → returns (normalized_proj, logits)
        """
        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map)
        feat = pooled.view(pooled.size(0), -1)  # [B, 512]

        # classification head
        x_cls = F.relu(self.fc1(feat))
        logits = self.fc2(x_cls)

        if return_feat:
            # projection for contrastive
            proj = self.proj_head(feat)
            proj = F.normalize(proj, dim=1)
            return proj, logits

        return logits
