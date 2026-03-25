import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict


class HookEngine:
    """
    Attaches forward and backward hooks to every Conv2d and Linear layer.
    Captures rich per-batch activation and gradient statistics.
    Call snapshot() to get per-layer averages across all recorded batches.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._data: Dict = defaultdict(lambda: {"activations": [], "gradients": []})
        self._handles = []
        self._register()

    def _register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._handles.append(
                    module.register_forward_hook(self._fwd_hook(name))
                )
                self._handles.append(
                    module.register_full_backward_hook(self._bwd_hook(name))
                )

    def _fwd_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                a = output.detach().float()
                flat = a.reshape(-1)

                pcts = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99],
                                                          device=flat.device))
                entry = {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "dead_fraction": (a == 0).float().mean().item(),
                    "abs_max": a.abs().max().item(),
                    "p1":  pcts[0].item(),
                    "p5":  pcts[1].item(),
                    "p50": pcts[2].item(),
                    "p95": pcts[3].item(),
                    "p99": pcts[4].item(),
                    "pos_fraction": (a > 0).float().mean().item(),
                    # Saturation: |a| > 10 flags runaway pre-activations
                    "sat_fraction": (flat.abs() > 10.0).float().mean().item(),
                }

                # Per-channel mean for Conv layers (B, C, H, W)
                if a.dim() == 4:
                    entry["channel_means"] = a.mean(dim=(0, 2, 3)).tolist()

                self._data[name]["activations"].append(entry)
        return hook

    def _bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach().float()
            flat = g.reshape(-1)
            abs_flat = flat.abs()

            pcts = torch.quantile(abs_flat, torch.tensor([0.50, 0.95, 0.99],
                                                          device=abs_flat.device))
            entry = {
                "mean": g.mean().item(),
                "std": g.std().item(),
                "norm": g.norm().item(),
                "abs_max": g.abs().max().item(),
                "abs_p50": pcts[0].item(),
                "abs_p95": pcts[1].item(),
                "abs_p99": pcts[2].item(),
                "near_zero_fraction": (abs_flat < 1e-7).float().mean().item(),
            }

            # Grad-to-weight ratio: ||∇W|| / ||W||
            if hasattr(module, "weight") and module.weight is not None:
                w_norm = module.weight.data.norm().item()
                if w_norm > 1e-10:
                    entry["grad_to_weight_ratio"] = g.norm().item() / w_norm

            self._data[name]["gradients"].append(entry)
        return hook

    def snapshot(self) -> Dict:
        """Average all recorded batch stats into one dict per layer."""
        summary = {}
        for layer, data in self._data.items():
            summary[layer] = {}
            for key in ("activations", "gradients"):
                entries = data[key]
                if not entries:
                    continue
                averaged = {}
                for k in entries[0]:
                    vals = [e[k] for e in entries if k in e]
                    if not vals:
                        continue
                    if isinstance(vals[0], list):
                        # Element-wise average for channel_means
                        n = len(vals[0])
                        averaged[k] = [
                            sum(v[i] for v in vals) / len(vals) for i in range(n)
                        ]
                    else:
                        averaged[k] = sum(vals) / len(vals)
                summary[layer][key] = averaged
        return summary

    def clear(self):
        for v in self._data.values():
            v["activations"].clear()
            v["gradients"].clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
