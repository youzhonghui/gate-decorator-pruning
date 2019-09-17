import torchvision
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

class Resnet50(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Resnet50, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=cfg.model.pretrained)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_classes != 1000:
            self.backbone.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_conv(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
