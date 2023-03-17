import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=6,
            kernel_size=4,
            stride=2
        )

    def forward(self, x):
        # Check the original matrix size and lag times!!!
        x = x.view(-1, 6, 20, 20)
        x = F.relu(self.conv1(x))

        return x.flatten(start_dim=1)


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 12, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Check the original matrix size and lag times!!!
        x = x.view(-1, 6, 20, 20)
        x = self.sequential(x)

        return x.flatten(start_dim=1)