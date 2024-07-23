import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def set_parameter_requires_grad(model, feature_extracting):
    """
    Set the `requires_grad` attribute of the parameters in the model based on whether feature extraction is being used.

    If feature extraction is being used, this function will freeze the parameters of the model to avoid updating their weights
    during training, allowing only the new layers you add to train. This is useful when adapting pre-trained models to new tasks
    by fine-tuning only the higher layers.

    Parameters:
        model (torch.nn.Module): The model whose parameters are to be frozen.
        feature_extracting (bool): A flag to determine whether to freeze the parameters.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ResNet50Model(nn.Module):
    """
    Custom implementation of ResNet50 for classification tasks with an option to use feature extraction and attention mechanism.

    This model builds upon the pre-trained ResNet50 architecture by replacing the final fully connected layer to adapt
    to the specified number of classes. Feature extraction can be toggled to use the pre-trained weights without fine-tuning
    the convolutions. The attention mechanism can be added optionally but is disabled by default for simplicity.

    Parameters:
        num_classes (int): Number of output classes for the classifier.
        feature_extract (bool, optional): Whether to use feature extraction. Defaults to True.
        use_attention (bool, optional): Indicates whether an attention layer should be used. Defaults to False.

    Attributes:
        resnet50 (torch.nn.Module): The ResNet50 backbone network.
        fc (torch.nn.Sequential): A sequence of layers forming the new fully connected head of the network.
        attention (torch.nn.Module, optional): An optional attention layer, not utilized by default.
    """

    def __init__(self, num_classes, feature_extract=True, use_attention=False):
        super(ResNet50Model, self).__init__()
        # Initialize ResNet50 with pre-trained weights if feature extraction is enabled, else no weights.
        self.resnet50 = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if feature_extract else None)
        # Freeze the parameters if feature extraction is enabled.
        set_parameter_requires_grad(self.resnet50, feature_extract)

        # Replace the fully connected layer of ResNet50.
        num_ftrs = self.resnet50.fc.in_features
        # Removing the original fully connected layer to replace it.
        self.resnet50.fc = nn.Identity()

        # Define a simpler fully connected layer sequence.
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Apply dropout to reduce overfitting.
            nn.Linear(num_ftrs, 256),  # Reduce dimension to 256.
            nn.ReLU(),  # Activation layer.
            # Final layer with `num_classes` outputs.
            nn.Linear(256, num_classes)
        )

        # Initialize the attention mechanism if specified.
        self.attention = None
        if use_attention:
            # Define the attention layer here.
            pass

    def forward(self, x):
        """
        Define the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.resnet50(x)  # Pass input through the ResNet50 backbone.
        if self.attention:
            x = self.attention(x)  # Apply attention mechanism if enabled.
        x = self.fc(x)  # Pass through the newly defined fully connected layer.
        return x
