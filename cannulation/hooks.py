import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict


class HookEngine:
    """
    Attaches forward and backward hooks to every Conv2d and Linear layer.
    Captures activation and gradient statistics at each batch.
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
                self._data[name]["activations"].append({
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "dead_fraction": (a == 0).float().mean().item(),
                    "abs_max": a.abs().max().item(),
                })
        return hook

    def _bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach().float()
            self._data[name]["gradients"].append({
                "mean": g.mean().item(),
                "std": g.std().item(),
                "norm": g.norm().item(),
                "abs_max": g.abs().max().item(),
            })
        return hook

    def snapshot(self) -> Dict:
        """Average all recorded batch stats into one dict per layer."""
        summary = {}
        for layer, data in self._data.items():
            summary[layer] = {}
            for key in ("activations", "gradients"):
                entries = data[key]
                if entries:
                    summary[layer][key] = {
                        k: sum(e[k] for e in entries) / len(entries)
                        for k in entries[0]
                    }
        return summary

    def clear(self):
        for v in self._data.values():
            v["activations"].clear()
            v["gradients"].clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
