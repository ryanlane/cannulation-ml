import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
}


class Visualizer:
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _chart_path(self, run_id: str) -> str:
        return os.path.join(self.output_dir, f"{run_id}_charts.json")

    def _base_layout(self, title: str, height: int = 360) -> Dict[str, Any]:
        return {
            "title": {"text": title, "x": 0.02, "xanchor": "left"},
            "height": height,
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#f8fafc",
            "margin": {"l": 56, "r": 32, "t": 56, "b": 48},
            "font": {"family": "ui-monospace, monospace", "size": 11},
            "legend": {"orientation": "h", "y": 1.12, "x": 0},
        }

    def _make_chart(
        self,
        slug: str,
        title: str,
        data: List[Dict[str, Any]],
        layout: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "slug": slug,
            "title": title,
            "data": data,
            "layout": layout,
            "config": config or DEFAULT_CONFIG,
        }

    def plot_training_curves(self, metrics: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        epochs = list(range(1, len(metrics["train_loss"]) + 1))
        layout = self._base_layout(f"Run {run_id} - Training Curves", height=380)
        layout.update({
            "xaxis": {"title": "Epoch", "dtick": 1, "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "yaxis": {"title": "Loss", "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "yaxis2": {"title": "Accuracy", "overlaying": "y", "side": "right", "range": [0, 1]},
            "hovermode": "x unified",
        })
        data = [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Train Loss",
                "x": epochs,
                "y": metrics["train_loss"],
                "line": {"color": "#2563eb", "width": 3},
                "marker": {"size": 6},
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Val Loss",
                "x": epochs,
                "y": metrics["val_loss"],
                "line": {"color": "#0f766e", "width": 3},
                "marker": {"size": 6},
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Train Acc",
                "x": epochs,
                "y": metrics["train_acc"],
                "yaxis": "y2",
                "line": {"color": "#ea580c", "width": 2, "dash": "dot"},
                "marker": {"size": 5},
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Val Acc",
                "x": epochs,
                "y": metrics["val_acc"],
                "yaxis": "y2",
                "line": {"color": "#be123c", "width": 2, "dash": "dot"},
                "marker": {"size": 5},
            },
        ]
        return self._make_chart("training_curves", "Training Curves", data, layout)

    def plot_gradient_flow(self, epoch_telemetry: List[Dict[str, Any]], run_id: str) -> Optional[Dict[str, Any]]:
        telemetry = epoch_telemetry[-1]
        layers = [layer for layer, values in telemetry.items() if values.get("gradients")]
        if not layers:
            return None

        norms = [float(telemetry[layer]["gradients"]["norm"]) for layer in layers]
        colors = [
            "#be123c" if norm < 1e-5 else "#ea580c" if norm > 10 else "#2563eb"
            for norm in norms
        ]
        statuses = [
            "vanishing" if norm < 1e-5 else "exploding" if norm > 10 else "healthy"
            for norm in norms
        ]
        layout = self._base_layout(f"Run {run_id} - Gradient Flow", height=max(360, 110 + len(layers) * 42))
        layout.update({
            "xaxis": {"title": "Gradient Norm (avg, final epoch)", "type": "log", "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "yaxis": {"title": "Layer", "automargin": True},
            "shapes": [
                {"type": "line", "x0": 1e-5, "x1": 1e-5, "y0": -0.5, "y1": len(layers) - 0.5, "line": {"color": "#be123c", "width": 2, "dash": "dash"}},
                {"type": "line", "x0": 10, "x1": 10, "y0": -0.5, "y1": len(layers) - 0.5, "line": {"color": "#ea580c", "width": 2, "dash": "dash"}},
            ],
            "annotations": [
                {"x": 1e-5, "y": 1.08, "xref": "x", "yref": "paper", "text": "Vanishing", "showarrow": False, "font": {"color": "#be123c"}},
                {"x": 10, "y": 1.08, "xref": "x", "yref": "paper", "text": "Exploding", "showarrow": False, "font": {"color": "#ea580c"}},
            ],
        })
        data = [{
            "type": "bar",
            "orientation": "h",
            "x": norms,
            "y": layers,
            "marker": {"color": colors},
            "customdata": statuses,
            "hovertemplate": "%{y}<br>norm=%{x:.6f}<br>state=%{customdata}<extra></extra>",
        }]
        return self._make_chart("gradient_flow", "Gradient Flow", data, layout)

    def plot_activation_health(self, epoch_telemetry: List[Dict[str, Any]], run_id: str) -> Optional[Dict[str, Any]]:
        layers = [layer for layer in epoch_telemetry[0] if epoch_telemetry[0][layer].get("activations")]
        if not layers:
            return None

        epochs = list(range(1, len(epoch_telemetry) + 1))
        data = []
        palette = ["#2563eb", "#0f766e", "#ea580c", "#7c3aed", "#be123c", "#0891b2"]
        for index, layer in enumerate(layers):
            values = [
                float(snapshot[layer]["activations"].get("dead_fraction", 0))
                if snapshot[layer].get("activations") else 0.0
                for snapshot in epoch_telemetry
            ]
            data.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": layer,
                "x": epochs,
                "y": values,
                "line": {"color": palette[index % len(palette)], "width": 2},
                "marker": {"size": 6},
            })

        layout = self._base_layout(f"Run {run_id} - Activation Health", height=380)
        layout.update({
            "xaxis": {"title": "Epoch", "dtick": 1, "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "yaxis": {"title": "Dead Neuron Fraction", "range": [0, 1], "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "hovermode": "x unified",
            "shapes": [
                {"type": "line", "x0": min(epochs), "x1": max(epochs), "y0": 0.5, "y1": 0.5, "line": {"color": "#be123c", "width": 2, "dash": "dash"}},
            ],
            "annotations": [
                {"x": max(epochs), "y": 0.53, "xref": "x", "yref": "y", "text": "Dead threshold (50%)", "showarrow": False, "font": {"color": "#be123c"}},
            ],
        })
        return self._make_chart("activation_health", "Activation Health", data, layout)

    def plot_weight_distributions(self, model, run_id: str) -> Optional[Dict[str, Any]]:
        named_params = [
            (name, param.detach().cpu().numpy().astype(float).ravel())
            for name, param in model.named_parameters()
            if "weight" in name
        ]
        if not named_params:
            return None

        traces: List[Dict[str, Any]] = []
        buttons: List[Dict[str, Any]] = []
        layer_summaries: List[Dict[str, Any]] = []
        for index, (name, weights) in enumerate(named_params):
            counts, edges = np.histogram(weights, bins=50)
            centers = ((edges[:-1] + edges[1:]) / 2).tolist()
            traces.append({
                "type": "bar",
                "name": name,
                "x": centers,
                "y": counts.tolist(),
                "marker": {"color": "#2563eb"},
                "visible": index == 0,
                "hovertemplate": "weight=%{x:.4f}<br>count=%{y}<extra></extra>",
            })
            summary = {
                "layer": name,
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            }
            layer_summaries.append(summary)
            buttons.append({
                "label": name,
                "method": "update",
                "args": [
                    {"visible": [trace_index == index for trace_index in range(len(named_params))]},
                    {
                        "annotations": [
                            {
                                "x": 0.99,
                                "y": 1.12,
                                "xref": "paper",
                                "yref": "paper",
                                "xanchor": "right",
                                "showarrow": False,
                                "align": "right",
                                "text": (
                                    f"mean={summary['mean']:.4f}<br>std={summary['std']:.4f}"
                                    f"<br>range=[{summary['min']:.4f}, {summary['max']:.4f}]"
                                ),
                            }
                        ]
                    },
                ],
            })

        first_summary = layer_summaries[0]
        layout = self._base_layout(f"Run {run_id} - Weight Distributions", height=420)
        layout.update({
            "xaxis": {"title": "Weight Value", "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "yaxis": {"title": "Count", "gridcolor": "rgba(148, 163, 184, 0.18)"},
            "updatemenus": [{
                "type": "dropdown",
                "x": 0,
                "y": 1.2,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }],
            "annotations": [{
                "x": 0.99,
                "y": 1.12,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "showarrow": False,
                "align": "right",
                "text": (
                    f"mean={first_summary['mean']:.4f}<br>std={first_summary['std']:.4f}"
                    f"<br>range=[{first_summary['min']:.4f}, {first_summary['max']:.4f}]"
                ),
            }],
        })
        return self._make_chart("weight_distributions", "Weight Distributions", traces, layout)

    def build_all(self, run_id: str, metrics: Dict[str, Any], model) -> List[Dict[str, Any]]:
        charts = [
            self.plot_training_curves(metrics, run_id),
            self.plot_weight_distributions(model, run_id),
            self.plot_gradient_flow(metrics["epoch_telemetry"], run_id),
            self.plot_activation_health(metrics["epoch_telemetry"], run_id),
        ]
        return [chart for chart in charts if chart is not None]

    def save_all(self, run_id: str, metrics: Dict[str, Any], model) -> str:
        payload = {
            "run_id": run_id,
            "charts": self.build_all(run_id, metrics, model),
        }
        path = self._chart_path(run_id)
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path
