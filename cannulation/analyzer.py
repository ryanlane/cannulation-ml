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

    def analyze(self, metrics: Dict, telemetry: Dict) -> List[Finding]:
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

        return findings
