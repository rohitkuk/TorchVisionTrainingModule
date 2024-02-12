import timm
import torch.nn as nn


class ResNextModel(nn.Module):
    """
    Model Class for ResNext Model Architectures
    """

    def __init__(self, num_classes=5, model_name='resnext50d_32x4d', pretrained=True):
        """
        Initialize the ResNextModel.

        Args:
            num_classes (int): Number of output classes.
            model_name (str): Name of the ResNext model architecture.
            pretrained (bool): Whether to load pretrained weights.

        """
        super(ResNextModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNextModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.model(x)
        return x


class ResNetModel(nn.Module):
    """
    Model Class for ResNet Models
    """

    def __init__(self, num_classes=5, model_name='resnet18', pretrained=True):
        """
        Initialize the ResNetModel.

        Args:
            num_classes (int): Number of output classes.
            model_name (str): Name of the ResNet model architecture.
            pretrained (bool): Whether to load pretrained weights.

        """
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNetModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.model(x)
        return x



class ConvNextModel(nn.Module):
    """
    Model Class for ResNext Model Architectures
    """

    def __init__(self, num_classes=5, model_name='convnext_small_in22k', pretrained=True):
        """
        Initialize the ConvNextModel.

        Args:
            num_classes (int): Number of output classes.
            model_name (str): Name of the ResNext model architecture.
            pretrained (bool): Whether to load pretrained weights.

        """
        super(ConvNextModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the ConvNextModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.model(x)
        return x