import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .datasets import DatasetInfo
from .hooks import HookEngine


class EarlyStopping:
    """
    Stops training when val_loss hasn't improved for `patience` consecutive epochs.
    Restores the best model weights on stop.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None
        self.stopped_epoch: Optional[int] = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


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
        patience = self.config.get("patience", 0)
        early_stopping = EarlyStopping(patience=patience) if patience > 0 else None

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

            if early_stopping is not None:
                if early_stopping.step(val_loss, self.model, epoch + 1):
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs, "
                          f"best val_loss={early_stopping.best_loss:.4f})")
                    early_stopping.restore(self.model)
                    metrics["stopped_early"] = True
                    metrics["best_epoch"] = (
                        epoch + 1 - patience
                    )
                    break

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
