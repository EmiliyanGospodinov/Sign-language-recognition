class CNN(nn.Module):

    def __init__(self, input_dims=(1,28,28)):

        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 7, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        "a trick to find out automatically the number of flattened features in the first fully connected layer"
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *(input_dims))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(num_features_before_fcnn, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
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