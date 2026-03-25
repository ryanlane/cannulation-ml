from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


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

    # Distribution-aware thresholds
    SAT_FRACTION_WARN = 0.20          # >20% activations |a| > 10 → saturating
    NEAR_ZERO_GRAD_WARN = 0.90        # >90% gradients near zero → vanishing
    GRAD_TO_WEIGHT_HIGH = 10.0        # grad/weight ratio > 10 → unstable
    GRAD_TO_WEIGHT_LOW = 1e-6         # grad/weight ratio < 1e-6 → no learning signal
    UPDATE_RATIO_HIGH = 0.10          # update/weight ratio > 10% → lr likely too high
    UPDATE_RATIO_LOW = 1e-5           # update/weight ratio < 1e-5 → lr likely too low

    # Embedding quality thresholds
    SILHOUETTE_POOR = 0.25
    SEPARATION_POOR = 1.5
    INTRINSIC_DIM_LOW = 0.10
    INTRINSIC_DIM_HIGH = 0.70

    def analyze(
        self,
        metrics: Dict,
        telemetry: Dict,
        emb_metrics: Optional[Dict] = None,
    ) -> List[Finding]:
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

                # Near-zero fraction: richer signal for vanishing gradients
                near_zero = grads.get("near_zero_fraction", 0.0)
                if near_zero > self.NEAR_ZERO_GRAD_WARN and norm >= self.VANISHING_THRESHOLD:
                    findings.append(Finding(
                        "warning", layer,
                        f"Most gradients near zero ({near_zero:.0%} < 1e-7)",
                        "Layer is barely learning — check learning rate and initialization",
                    ))

                # Grad-to-weight ratio
                gw = grads.get("grad_to_weight_ratio")
                if gw is not None:
                    if gw > self.GRAD_TO_WEIGHT_HIGH:
                        findings.append(Finding(
                            "warning", layer,
                            f"High gradient-to-weight ratio ({gw:.2f}) — gradients dominate weights",
                            "Reduce learning rate or add gradient clipping",
                        ))
                    elif gw < self.GRAD_TO_WEIGHT_LOW:
                        findings.append(Finding(
                            "info", layer,
                            f"Very low gradient-to-weight ratio ({gw:.2e}) — negligible signal",
                            "Layer may have stopped learning — check upstream gradient flow",
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

                sat = acts.get("sat_fraction", 0.0)
                if sat > self.SAT_FRACTION_WARN:
                    findings.append(Finding(
                        "warning", layer,
                        f"Saturated activations ({sat:.0%} with |a| > 10)",
                        "Activations are extremely large — reduce learning rate or add normalization",
                    ))

        # Update-to-weight ratio check (uses last epoch's ratios)
        update_ratios = metrics.get("epoch_update_ratios", [{}])
        if update_ratios:
            last_ratios = update_ratios[-1]
            high_layers = [n for n, r in last_ratios.items() if r > self.UPDATE_RATIO_HIGH]
            low_layers = [n for n, r in last_ratios.items() if r < self.UPDATE_RATIO_LOW]
            if high_layers:
                findings.append(Finding(
                    "warning", "optimizer",
                    f"Large weight updates detected in {len(high_layers)} layer(s) (ratio > 10%)",
                    "Learning rate may be too high — consider reducing it",
                ))
            elif low_layers and len(low_layers) >= len(last_ratios) // 2:
                findings.append(Finding(
                    "info", "optimizer",
                    f"Very small weight updates in {len(low_layers)} layer(s) (ratio < 1e-5)",
                    "Learning rate may be too low — model is not changing much each step",
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
