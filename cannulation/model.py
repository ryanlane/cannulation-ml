import torch.nn as nn
import torch.nn.functional as F


class CannulationCNN(nn.Module):
    def __init__(self, conv_channels=(32, 64), fc_size=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # 28x28 -> pool -> 14x14 -> pool -> 7x7
        fc_input = conv_channels[1] * 7 * 7
        self.fc1 = nn.Linear(fc_input, fc_size)
        self.fc2 = nn.Linear(fc_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
