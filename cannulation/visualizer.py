import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Optional


class Visualizer:
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, name: str, run_id: str) -> str:
        path = os.path.join(self.output_dir, f"{run_id}_{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        return path

    def plot_training_curves(self, metrics: Dict, run_id: str) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(metrics["train_loss"]) + 1)

        ax1.plot(epochs, metrics["train_loss"], label="Train")
        ax1.plot(epochs, metrics["val_loss"], label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, metrics["train_acc"], label="Train")
        ax2.plot(epochs, metrics["val_acc"], label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Run {run_id} — Training Curves")
        return self._save("training_curves", run_id)

    def plot_gradient_flow(self, epoch_telemetry: List[Dict], run_id: str) -> Optional[str]:
        telemetry = epoch_telemetry[-1]
        layers = [k for k, v in telemetry.items() if v.get("gradients")]
        if not layers:
            return None

        norms = [telemetry[l]["gradients"]["norm"] for l in layers]
        colors = [
            "crimson" if n < 1e-5 else "darkorange" if n > 10 else "steelblue"
            for n in norms
        ]

        fig, ax = plt.subplots(figsize=(10, max(3, len(layers) * 0.7)))
        ax.barh(layers, norms, color=colors)
        ax.axvline(1e-5, color="crimson", linestyle="--", alpha=0.6, label="Vanishing (<1e-5)")
        ax.axvline(10, color="darkorange", linestyle="--", alpha=0.6, label="Exploding (>10)")
        ax.set_xlabel("Gradient Norm (avg, final epoch)")
        ax.set_title(f"Run {run_id} — Gradient Flow")
        ax.legend(); ax.grid(True, alpha=0.3, axis="x")
        return self._save("gradient_flow", run_id)

    def plot_activation_health(self, epoch_telemetry: List[Dict], run_id: str) -> Optional[str]:
        layers = [k for k in epoch_telemetry[0] if epoch_telemetry[0][k].get("activations")]
        if not layers:
            return None

        dead_over_time = {
            l: [snap[l]["activations"].get("dead_fraction", 0)
                if snap[l].get("activations") else 0
                for snap in epoch_telemetry]
            for l in layers
        }

        fig, ax = plt.subplots(figsize=(10, 4))
        for l in layers:
            ax.plot(range(1, len(epoch_telemetry) + 1), dead_over_time[l], marker="o", label=l)
        ax.axhline(0.5, color="crimson", linestyle="--", alpha=0.5, label="Dead threshold (50%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dead Neuron Fraction")
        ax.set_ylim(0, 1)
        ax.set_title(f"Run {run_id} — Activation Health")
        ax.legend(); ax.grid(True, alpha=0.3)
        return self._save("activation_health", run_id)

    def plot_weight_distributions(self, model, run_id: str) -> str:
        import torch.nn as nn
        named_params = [
            (name, param.detach().cpu().numpy().flatten())
            for name, param in model.named_parameters()
            if "weight" in name
        ]
        n = len(named_params)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]

        for ax, (name, weights) in zip(axes, named_params):
            ax.hist(weights, bins=50, color="steelblue", alpha=0.8)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("Weight value")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Run {run_id} — Weight Distributions")
        fig.tight_layout()
        return self._save("weight_distributions", run_id)

    def plot_tsne(self, embeddings: np.ndarray, labels: np.ndarray, run_id: str) -> str:
        print("  Computing t-SNE embeddings...")
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Digit class")
        ax.set_title(f"Run {run_id} — t-SNE of Learned Embeddings")
        ax.axis("off")
        return self._save("tsne", run_id)

    def plot_all(self, run_id: str, metrics: Dict, model,
                 embeddings: np.ndarray, labels: np.ndarray) -> List[str]:
        paths = [
            self.plot_training_curves(metrics, run_id),
            self.plot_weight_distributions(model, run_id),
        ]
        p = self.plot_gradient_flow(metrics["epoch_telemetry"], run_id)
        if p:
            paths.append(p)
        p = self.plot_activation_health(metrics["epoch_telemetry"], run_id)
        if p:
            paths.append(p)
        paths.append(self.plot_tsne(embeddings, labels, run_id))
        return paths
