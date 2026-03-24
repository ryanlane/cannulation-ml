import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, Any

from .hooks import HookEngine


def get_dataloaders(batch_size: int = 64, data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], hooks: HookEngine):
        self.model = model
        self.config = config
        self.hooks = hooks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs: int = 5) -> Dict[str, list]:
        train_loader, test_loader = get_dataloaders(self.config["batch_size"])

        metrics: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_telemetry": [],
        }

        for epoch in range(epochs):
            self.model.train()
            self.hooks.clear()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)

            metrics["train_loss"].append(total_loss / total)
            metrics["train_acc"].append(correct / total)
            metrics["epoch_telemetry"].append(self.hooks.snapshot())

            val_loss, val_acc = self._evaluate(test_loader)
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch+1}/{epochs} | "
                f"loss {metrics['train_loss'][-1]:.4f} | "
                f"acc {metrics['train_acc'][-1]:.3f} | "
                f"val_loss {val_loss:.4f} | "
                f"val_acc {val_acc:.3f}"
            )

        return metrics

    def _evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        return total_loss / total, correct / total

    def get_embeddings(self, n_samples: int = 500):
        """Extract fc1 activations for t-SNE visualization."""
        _, test_loader = get_dataloaders(self.config["batch_size"])
        embeddings, label_list = [], []
        captured = {}

        handle = self.model.fc1.register_forward_hook(
            lambda m, i, o: captured.__setitem__("emb", o.detach().cpu())
        )
        self.model.eval()

        count = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                self.model(images)
                embeddings.append(captured["emb"])
                label_list.append(labels)
                count += len(labels)
                if count >= n_samples:
                    break

        handle.remove()
        emb = torch.cat(embeddings)[:n_samples].numpy()
        lbl = torch.cat(label_list)[:n_samples].numpy()
        return emb, lbl
