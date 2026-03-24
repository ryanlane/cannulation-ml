from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class Finding:
    severity: str   # 'info' | 'warning' | 'critical'
    layer: str
    issue: str
    suggestion: str

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return f"[{self.severity.upper():8}] {self.layer}: {self.issue}  →  {self.suggestion}"


class Analyzer:
    VANISHING_THRESHOLD = 1e-5
    EXPLODING_THRESHOLD = 10.0
    DEAD_NEURON_THRESHOLD = 0.5
    OVERFIT_GAP = 0.05

    # Embedding quality thresholds
    SILHOUETTE_POOR = 0.25
    SEPARATION_POOR = 1.5
    INTRINSIC_DIM_LOW = 0.10   # fraction of embedding_dim — below this is over-parameterized
    INTRINSIC_DIM_HIGH = 0.70  # fraction of embedding_dim — above this, fc layer too small

    def analyze(self, metrics: Dict, telemetry: Dict,
                emb_metrics: Dict = None) -> List[Finding]:
        findings = []

        for layer, data in telemetry.items():
            grads = data.get("gradients")
            if grads:
                norm = grads["norm"]
                if norm < self.VANISHING_THRESHOLD:
                    findings.append(Finding(
                        "critical", layer,
                        f"Vanishing gradient (norm={norm:.2e})",
                        "Increase learning rate or add batch normalization",
                    ))
                elif norm > self.EXPLODING_THRESHOLD:
                    findings.append(Finding(
                        "critical", layer,
                        f"Exploding gradient (norm={norm:.2f})",
                        "Decrease learning rate or add gradient clipping",
                    ))

            acts = data.get("activations")
            if acts:
                dead = acts["dead_fraction"]
                if dead > self.DEAD_NEURON_THRESHOLD:
                    findings.append(Finding(
                        "warning", layer,
                        f"Dead neurons ({dead:.0%} zeros)",
                        "Reduce dropout or check weight initialization",
                    ))

        # Convergence check
        if len(metrics["val_acc"]) >= 2:
            delta = metrics["val_acc"][-1] - metrics["val_acc"][-2]
            if abs(delta) < 0.001:
                findings.append(Finding(
                    "info", "training",
                    f"Slow convergence (Δval_acc={delta:+.4f})",
                    "Consider increasing learning rate or changing architecture",
                ))

        # Overfitting check
        if metrics["train_acc"] and metrics["val_acc"]:
            gap = metrics["train_acc"][-1] - metrics["val_acc"][-1]
            if gap > self.OVERFIT_GAP:
                findings.append(Finding(
                    "warning", "training",
                    f"Overfitting (train-val gap={gap:.3f})",
                    "Increase dropout or reduce model capacity",
                ))

        # Embedding quality checks
        if emb_metrics:
            sil = emb_metrics["silhouette"]
            sep = emb_metrics["separation_ratio"]
            dims_90 = emb_metrics["dims_for_90pct_var"]
            emb_dim = emb_metrics["embedding_dim"]
            worst = emb_metrics["worst_separated_pair"]

            if sil < self.SILHOUETTE_POOR:
                findings.append(Finding(
                    "warning", "embeddings",
                    f"Poor class separation in representation space (silhouette={sil:.3f})",
                    "Increase fc_size or conv_channels to give the model more representational capacity",
                ))
            elif sil < 0.45:
                findings.append(Finding(
                    "info", "embeddings",
                    f"Moderate class separation (silhouette={sil:.3f})",
                    "Model is learning reasonable representations but has room to improve",
                ))

            if sep < self.SEPARATION_POOR:
                findings.append(Finding(
                    "warning", "embeddings",
                    f"Classes overlapping in embedding space (separation ratio={sep:.2f})",
                    f"Classes {worst[0]} and {worst[1]} are hardest to separate — "
                    "consider larger conv layers or more training epochs",
                ))

            intrinsic_fraction = dims_90 / emb_dim
            if intrinsic_fraction < self.INTRINSIC_DIM_LOW:
                findings.append(Finding(
                    "info", "embeddings",
                    f"Low intrinsic dimensionality ({dims_90} dims explain 90% of variance "
                    f"in {emb_dim}-dim space)",
                    "Embedding layer is over-parameterized — reducing fc_size would cut "
                    "compute with minimal accuracy loss",
                ))
            elif intrinsic_fraction > self.INTRINSIC_DIM_HIGH:
                findings.append(Finding(
                    "warning", "embeddings",
                    f"High intrinsic dimensionality ({dims_90} of {emb_dim} dims needed for 90% variance)",
                    "fc_size may be too small — the model is compressing too aggressively",
                ))

        return findings
