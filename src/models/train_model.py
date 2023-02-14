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


#########################################################################################


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


#########################################################################################


class Resnet3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_stack1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.Dropout(0.3),
                    nn.Tanh(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.Dropout(0.3),
                    nn.Tanh(),
                ) for _ in range(10)
        ])
        self.maxpool1 = nn.MaxPool2d((4, 1))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_stack2 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.Dropout(0.3),
                    nn.Tanh(),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.Dropout(0.3),
                    nn.Tanh(),
                ) for _ in range(5)
        ])
        self.maxpool2 = nn.MaxPool2d((1, 3))
        self.flatten = nn.Flatten()

        self.pitch_layer = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(1024, 129),
            nn.LogSoftmax(dim=1)
        )

        self.velocity_layer = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(2048, 1),
        )
        self.count_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(1024, 1),
        )
        self.base = nn.ModuleList([
            self.conv_layer1,
            self.conv_stack1,
            self.maxpool1,
            self.conv_layer2,
            self.conv_stack2,
            self.maxpool2,
            self.pitch_layer
        ])

    def forward(self, x: torch.Tensor):
        x = self.conv_layer1(x)
        for layer in self.conv_stack1:
            x = layer(x) + x
        x = self.maxpool1(x)

        x = self.conv_layer2(x)
        for layer in self.conv_stack2:
            x = layer(x) + x
        x = self.maxpool2(x)
        x = self.flatten(x)

        pitch = self.pitch_layer(x)
        # with torch.no_grad():
        velocity = self.velocity_layer(x.clone().detach())
        notes_count = self.count_layer(x.clone().detach())
        # velocity = torch.full((x.shape[0], 1), 80, device=device, dtype=float)
        # notes_count = torch.full((x.shape[0], 1), 5, device=device, dtype=float)
        return pitch, velocity, notes_count


#########################################################################################


class DummyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.maxpool1 = nn.MaxPool2d((4, 1))

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        self.conv_layer6 = nn.Sequential(
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
        x = self.conv_layer3(x)
        x = self.conv_layer4(x) + x
        x = self.maxpool1(x)

        x = self.conv_layer5(x)
        x = self.conv_layer6(x) + x
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.linear_layer(x)
        pitch = self.pitch_layer(x)
        #         velocity = self.velocity_layer(x)
        #         notes_count = self.count_layer(x)
        velocity = torch.full((x.shape[0], 1), 80, device=device, dtype=float)
        notes_count = torch.full((x.shape[0], 1), 5, device=device, dtype=float)
        return pitch, velocity, notes_count
