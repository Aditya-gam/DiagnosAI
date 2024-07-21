import torch
import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ResNet50Model(nn.Module):
    def __init__(self, num_classes, feature_extract=True, use_attention=False):
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        set_parameter_requires_grad(self.resnet50, feature_extract)

        # Modify the fully connected layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # Remove the original fully connected layer

        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

        # Optional: Add an attention module
        if use_attention:
            self.attention = SelfAttention(num_ftrs)
        else:
            self.attention = None

    def forward(self, x):
        x = self.resnet50(x)

        if self.attention:
            x = self.attention(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    """ A simple self-attention module """

    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) * (1. / Q.size(-1)**0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        return attended + x  # Skip connection
