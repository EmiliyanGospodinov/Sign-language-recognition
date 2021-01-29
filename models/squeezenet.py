import torch.nn as nn
from torchvision import models


def initialize_model():
    """
    Initialize a Squeezenet1_0 model from pytorch,
    not pretrained and adjusted to work on our dataset
    this model will be basically use for comparison in the final report

    Returns
    -------
    Squeezenet: Cnn model
    """
    model = models.squeezenet1_0(pretrained=False)
    model.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
    model.classifier[1] = nn.Conv2d(512, 25, kernel_size=(1, 1), stride=(1, 1))

    return model