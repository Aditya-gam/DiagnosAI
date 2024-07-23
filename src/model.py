import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, feature_extract=True, use_attention=False):
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if feature_extract else None)
        set_parameter_requires_grad(self.resnet50, feature_extract)

        # Modify the fully connected layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # Remove the original fully connected layer

        # Simpler and smaller fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),  # Reduced from 512 to 256
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Remove attention to simplify the model
        self.attention = None

    def forward(self, x):
        x = self.resnet50(x)
        if self.attention:
            x = self.attention(x)
        x = self.fc(x)
        return x
