import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator

class CNN(nn.Module):
    """
    Convolutional neural network that was found to generalize best
    not only on our dataset, but also on 
    "https://www.kaggle.com/grassknoted/asl-alphabet" and "https://www.kaggle.com/ayuraj/asl-dataset"

    Returns
    -------
    CNN: convolutional neural network    
    """
    def __init__(self, input_dims=(1,28,28)):

        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size = 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        "a trick to find out automatically the number of flattened features for the first fully connected layer"
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *(input_dims))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(num_features_before_fcnn, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 25),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        size = x.size(0)
        out = self.feature_extractor(x)
        out = out.view(size, -1)
        out = self.classifier(out)
        return out
