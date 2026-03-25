import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List
from torch.utils.data import DataLoader

from .datasets import DatasetInfo


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    dataset_info: DatasetInfo,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run full evaluation on the validation set.
    Returns confusion matrix, per-class metrics, calibration, and hardest examples.
    Only meaningful for classification tasks.
    """
    if dataset_info.task_type != "classification":
        return {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_labels, all_preds, all_probs, all_losses = [], [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            losses = criterion(outputs, labels)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_losses.extend(losses.cpu().numpy().tolist())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    losses_arr = np.array(all_losses)

    num_classes = dataset_info.num_classes
    confusion = _confusion_matrix(labels_arr, preds_arr, num_classes)
    per_class = _per_class_metrics(confusion, num_classes, dataset_info.class_names)
    ece = _expected_calibration_error(probs_arr, labels_arr)

    # Top-20 highest-loss samples
    hardest_idx = np.argsort(losses_arr)[-20:][::-1].tolist()
    hardest_examples = [
        {
            "idx": int(i),
            "loss": round(float(losses_arr[i]), 4),
            "label": int(labels_arr[i]),
            "pred": int(preds_arr[i]),
            "confidence": round(float(probs_arr[i][preds_arr[i]]), 4),
        }
        for i in hardest_idx
    ]

    # Top-20 most uncertain (lowest max-prob)
    max_probs = probs_arr.max(axis=1)
    uncertain_idx = np.argsort(max_probs)[:20].tolist()
    uncertain_examples = [
        {
            "idx": int(i),
            "label": int(labels_arr[i]),
            "pred": int(preds_arr[i]),
            "confidence": round(float(max_probs[i]), 4),
        }
        for i in uncertain_idx
    ]

    return {
        "overall_accuracy": round(float((labels_arr == preds_arr).mean()), 4),
        "num_classes": num_classes,
        "confusion_matrix": confusion.tolist(),
        "per_class_metrics": per_class,
        "ece": ece,
        "hardest_examples": hardest_examples,
        "uncertain_examples": uncertain_examples,
        "hardest_class_pairs": _hardest_class_pairs(confusion, top_n=5),
    }


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t)][int(p)] += 1
    return cm


def _per_class_metrics(
    confusion: np.ndarray, num_classes: int, class_names: List[str] = None
) -> List[Dict[str, Any]]:
    result = []
    for i in range(num_classes):
        tp = int(confusion[i][i])
        fp = int(confusion[:, i].sum()) - tp
        fn = int(confusion[i, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(confusion[i, :].sum())
        result.append({
            "class": i,
            "name": class_names[i] if class_names and i < len(class_names) else str(i),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        })
    return result


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = max_probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return round(float(ece), 4)


def _hardest_class_pairs(confusion: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
    n = confusion.shape[0]
    pairs = [
        {"true": i, "pred": j, "count": int(confusion[i][j])}
        for i in range(n) for j in range(n)
        if i != j and confusion[i][j] > 0
    ]
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]
