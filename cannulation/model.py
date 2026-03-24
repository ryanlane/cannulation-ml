import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import DatasetInfo


class CannulationCNN(nn.Module):
    def __init__(
        self,
        conv_channels=(32, 64),
        fc_size=128,
        dropout=0.3,
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            fc_input = self._forward_features(dummy).flatten(1).shape[1]

        self.fc1 = nn.Linear(fc_input, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.embedding_layer = self.fc1

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, fc_size: int = 128, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        hidden_size = max(fc_size // 2, 32)
        self.fc1 = nn.Linear(input_dim, fc_size)
        self.fc2 = nn.Linear(fc_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.embedding_layer = self.fc1

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.out(x)


def build_model(config: dict, dataset_info: DatasetInfo) -> nn.Module:
    if dataset_info.dataset_type == "tabular":
        if dataset_info.input_dim is None or dataset_info.num_classes is None:
            raise ValueError("Tabular datasets require input_dim and num_classes")
        return TabularMLP(
            input_dim=dataset_info.input_dim,
            fc_size=config["fc_size"],
            dropout=config["dropout"],
            num_classes=dataset_info.num_classes,
        )

    if dataset_info.input_shape is None or dataset_info.num_classes is None:
        raise ValueError("Image datasets require input_shape and num_classes")

    channels, height, width = dataset_info.input_shape
    return CannulationCNN(
        conv_channels=tuple(config["conv_channels"]),
        fc_size=config["fc_size"],
        dropout=config["dropout"],
        input_channels=channels,
        input_size=(height, width),
        num_classes=dataset_info.num_classes,
    )
