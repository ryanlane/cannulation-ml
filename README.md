# Cannulation ML

> *Cannulation: the insertion of a tube into a body cavity to observe or extract.*

A machine learning framework that exposes the internal learning process of neural networks — activations, gradients, weight distributions, embedding geometry, and convergence behavior — then feeds those observations back in to automatically improve the next run.

The goal is to make ML more efficient by surfacing what's actually happening inside the black box and letting the system self-correct across iterations.

---

## What It Does

Most training scripts tell you loss and accuracy. Cannulation ML tells you *why* those numbers look the way they do:

- Which layers have dead neurons and where gradients are vanishing or exploding
- Whether the model is overfitting or converging slowly
- How well the model has learned to separate classes in embedding space
- Whether the architecture is over- or under-parameterized for the data
- Which classes the model struggles with most

After each run, an analyzer flags problems and a tuner proposes config changes for the next run automatically — adjusting not just learning rate and dropout, but also architecture (conv channels, fc layer size) based on embedding geometry.

---

## Architecture

```
datasets.py          Universal dataset loader: MNIST, CSV, image folder, HuggingFace
model.py             Configurable CNN (image) and MLP (tabular), auto-sized to dataset
hooks.py             Forward/backward hooks on every layer — activation + gradient stats
trainer.py           Training loop that streams telemetry through hooks
analyzer.py          Emits findings from telemetry and embedding metrics
embedding_metrics.py Silhouette score, separation ratio, intrinsic dimensionality,
                     per-class compactness — computed on raw pre-t-SNE embeddings
visualizer.py        Interactive Plotly charts saved as JSON per run
tuner.py             Maps findings to config adjustments for the next run
runner.py            CLI entry point — orchestrates the full pipeline
web.py               FastAPI dashboard — run control, live logs, charts, embeddings
```

Each run writes a JSON record to `experiments/` containing metrics, analyzer findings, embedding quality metrics, and the tuner's suggestion for next run. Interactive charts and raw embeddings are saved alongside.

---

## Quickstart

```bash
# Install dependencies
uv sync

# First run — MNIST, default config, 5 epochs
uv run cannulation

# Continue from last run's auto-tuned config
uv run cannulation --iterate

# Chain 3 self-improving runs back to back
uv run cannulation --runs 3 --iterate

# Use a custom dataset
uv run cannulation --data /path/to/data.csv --target-col label
uv run cannulation --data /path/to/image-folder
uv run cannulation --data mnist  # explicit

# Start the web dashboard
uv run cannulation-web
# → http://localhost:8000
```

---

## Web Dashboard

Launch with `uv run cannulation-web` and open `http://localhost:8000` (or replace `localhost` with the machine hostname for remote access).

**Index page** — all runs in a table with accuracy, findings badge, config, and elapsed time. Launch new runs directly from the browser with a full config form. Live training log streams as the model trains.

**Run detail page:**
- Config and analyzer findings side by side
- Tuner suggestion — what will change next run and why
- Epoch metrics table with Δ val acc column
- Interactive Plotly charts: training curves, gradient flow, activation health, weight distributions
- **Embedding Quality card** — silhouette score, separation ratio, intrinsic dimensionality, per-class compactness bar chart
- **Interactive embedding visualization** — 2D/3D/4D/5D t-SNE with three view modes:
  - **Scatter** — spatial cluster view, fully rotatable in 3D
  - **Parallel coordinates** — all N dimensions as parallel axes, brush to filter
  - **Matrix (SPLOM)** — pairwise scatter for all dimension combinations

---

## Analyzer Findings

| Finding | Trigger | Tuner Response |
|---|---|---|
| Vanishing gradient | Layer grad norm < 1e-5 | Increase learning rate |
| Exploding gradient | Layer grad norm > 10 | Decrease learning rate |
| Dead neurons | >50% zero activations in a layer | Reduce dropout |
| Overfitting | Train-val accuracy gap > 5% | Increase dropout |
| Slow convergence | Val accuracy delta < 0.001 | Increase learning rate |
| Poor class separation | Silhouette score < 0.25 | Increase conv_channels + fc_size |
| Classes overlapping | Separation ratio < 1.5 | Increase conv_channels + fc_size |
| Over-parameterized | dims for 90% variance < 10% of fc_size | Reduce fc_size |
| Under-parameterized | dims for 90% variance > 70% of fc_size | Increase fc_size |

---

## Experiment Record

Each run saves `experiments/<run_id>.json`:

```json
{
  "run_id": "20260324_134241",
  "config": { "lr": 0.001, "dropout": 0.3, "conv_channels": [32, 64], "fc_size": 128, ... },
  "dataset": { "source": "mnist", "dataset_type": "image", "task_type": "classification", ... },
  "metrics": { "train_acc": [...], "val_acc": [...], "train_loss": [...], "val_loss": [...] },
  "embedding_metrics": {
    "silhouette": 0.412,
    "separation_ratio": 3.21,
    "dims_for_90pct_var": 14,
    "dims_for_99pct_var": 38,
    "embedding_dim": 128,
    "worst_separated_pair": [4, 9, 12.3]
  },
  "findings": [...],
  "next_config": { "lr": 0.001, "dropout": 0.3, ... },
  "elapsed_seconds": 82.5
}
```

Alongside each record:
- `<run_id>_charts.json` — interactive Plotly chart payloads
- `<run_id>_embeddings.json` — raw fc1 activations for t-SNE (computed on demand, cached per dimension count)

---

## Dataset Support

| Source | How to specify |
|---|---|
| MNIST (default) | omit `--data`, or `--data mnist` |
| CSV / tabular | `--data /path/to/file.csv --target-col <column>` |
| Image folder | `--data /path/to/folder` (structured as `class/image.jpg`) |
| HuggingFace | `--data <dataset-name> --target-col <column>` |

---

## Project Structure

```
cannulation-ml/
├── cannulation/
│   ├── __init__.py
│   ├── __main__.py
│   ├── datasets.py          # Dataset loading and normalization
│   ├── model.py             # CannulationCNN, TabularMLP, build_model()
│   ├── hooks.py             # PyTorch forward/backward hook engine
│   ├── trainer.py           # Training loop + embedding extraction
│   ├── analyzer.py          # Findings from telemetry and embedding metrics
│   ├── embedding_metrics.py # Cluster quality and intrinsic dimensionality
│   ├── visualizer.py        # Interactive Plotly chart builder
│   ├── tuner.py             # Config adjustment from findings
│   ├── runner.py            # CLI orchestrator
│   ├── web.py               # FastAPI dashboard server
│   └── templates/
│       ├── base.html
│       ├── index.html
│       └── run.html
├── experiments/             # Run records, chart payloads, embedding cache
├── data/                    # Dataset download cache
├── PLAN.md                  # Full feature roadmap
├── pyproject.toml
└── uv.lock
```
