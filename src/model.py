import torch
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, feature_extract=True):
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        set_parameter_requires_grad(self.resnet50, feature_extract)
        
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)  # Replace the fully connected layer

    def forward(self, x):
        x = self.resnet50(x)
        return x
