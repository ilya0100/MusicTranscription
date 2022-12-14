import torch as nn


class DummyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.maxpool1 = nn.MaxPool2d((4, 1))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.maxpool2 = nn.MaxPool2d((1, 3))
        self.flatten = nn.Flatten()

        self.linear_layer = nn.Sequential(
            nn.Linear(16384, 8000),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(8000, 129),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x) + x
        x = self.maxpool1(x)

        x = self.conv_layer3(x)
        x = self.conv_layer4(x) + x
        x = self.conv_layer5(x) + x
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.linear_layer(x)
        return x, 90
