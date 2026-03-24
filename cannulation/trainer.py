import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

from .datasets import DatasetInfo
from .hooks import HookEngine


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        hooks: HookEngine,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_info: DatasetInfo,
    ):
        self.model = model
        self.config = config
        self.hooks = hooks
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_info = dataset_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs: int = 5) -> Dict[str, list]:
        metrics: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_telemetry": [],
        }

        for epoch in range(epochs):
            self.model.train()
            self.hooks.clear()
            total_loss, correct, total = 0.0, 0, 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)

            metrics["train_loss"].append(total_loss / total)
            metrics["train_acc"].append(correct / total)
            metrics["epoch_telemetry"].append(self.hooks.snapshot())

            val_loss, val_acc = self._evaluate(self.val_loader)
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
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        return total_loss / total, correct / total

    def get_embeddings(self, n_samples: int = 500):
        """Extract embedding layer activations for t-SNE visualization."""
        if self.dataset_info.task_type != "classification":
            return None, None

        embedding_layer = getattr(self.model, "embedding_layer", None)
        if embedding_layer is None:
            return None, None

        embeddings, label_list = [], []
        captured = {}

        handle = embedding_layer.register_forward_hook(
            lambda m, i, o: captured.__setitem__("emb", o.detach().cpu())
        )
        self.model.eval()

        count = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                self.model(inputs)
                embeddings.append(captured["emb"])
                label_list.append(labels.detach().cpu())
                count += len(labels)
                if count >= n_samples:
                    break

        handle.remove()
        if not embeddings:
            return None, None

        emb = torch.cat(embeddings)[:n_samples].numpy()
        lbl = torch.cat(label_list)[:n_samples].numpy()
        return emb, lbl
