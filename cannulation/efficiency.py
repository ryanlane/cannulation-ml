import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from .datasets import DatasetInfo


def measure_model(model: nn.Module, dataset_info: DatasetInfo) -> Dict[str, Any]:
    """Count parameters, model size, and optionally FLOPs for one forward pass."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = round((param_bytes + buffer_bytes) / (1024 ** 2), 3)

    flops: Optional[int] = _count_flops(model, dataset_info)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "flops_per_sample": flops,
    }


def add_efficiency_ratios(efficiency: Dict[str, Any], val_acc: float) -> Dict[str, Any]:
    """Append accuracy-per-compute ratios after training is complete."""
    result = dict(efficiency)
    params_m = efficiency["total_params"] / 1e6
    if params_m > 0:
        result["val_acc_per_million_params"] = round(val_acc / params_m, 4)
    flops = efficiency.get("flops_per_sample")
    if flops and flops > 0:
        flops_b = flops / 1e9
        result["val_acc_per_billion_flops"] = round(val_acc / flops_b, 4) if flops_b > 0 else None
    return result


def _count_flops(model: nn.Module, dataset_info: DatasetInfo) -> Optional[int]:
    input_shape = _get_input_shape(dataset_info)
    if input_shape is None:
        return None
    try:
        from torchinfo import summary
        device = next(model.parameters()).device
        info = summary(model, input_size=(1, *input_shape), device=device, verbose=0)
        return int(info.total_mult_adds)
    except Exception:
        return None


def _get_input_shape(dataset_info: DatasetInfo) -> Optional[tuple]:
    if dataset_info.dataset_type == "image" and dataset_info.input_shape:
        return dataset_info.input_shape
    if dataset_info.dataset_type == "tabular" and dataset_info.input_dim:
        return (dataset_info.input_dim,)
    return None
