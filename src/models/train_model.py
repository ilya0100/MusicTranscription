import torch
from torch import nn


gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")


class Resnet1(nn.Module):
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


class Resnet2(nn.Module):
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
            nn.Linear(16384, 8192),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.pitch_layer = nn.Sequential(nn.Linear(8192, 129), nn.LogSoftmax(dim=1))

    #         self.velocity_layer = nn.Sequential(
    #             nn.Linear(2048, 1024),
    #             nn.Dropout(0.3),
    #             nn.Tanh(),
    #             nn.Linear(1024, 1),
    #         )
    #         self.count_layer = nn.Linear(2048, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x) + x
        x = self.maxpool1(x)

        x = self.conv_layer3(x)
        x = self.conv_layer4(x) + x
        x = self.conv_layer5(x) + x
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.linear_layer(x)
        pitch = self.pitch_layer(x)
        #         velocity = self.velocity_layer(x)
        #         notes_count = self.count_layer(x)
        velocity = torch.full((x.shape[0], 1), 80, device=device, dtype=float)
        notes_count = torch.full((x.shape[0], 1), 5, device=device, dtype=float)
        return pitch, velocity, notes_count
