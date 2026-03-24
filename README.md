# Cannulation ML

> *Cannulation: the insertion of a tube into a body cavity to observe or extract.*

A machine learning framework that exposes the internal learning process of neural networks — activations, gradients, weight distributions, and convergence behavior — then feeds those observations back in to automatically improve the next run.

The goal is to make ML more efficient by surfacing what's actually happening inside the black box and letting the system self-correct across iterations.

---

## What It Does

Most training scripts tell you loss and accuracy. Cannulation ML tells you *why* those numbers look the way they do:

- Which layers have dead neurons
- Where gradients are vanishing or exploding
- Whether the model is overfitting or converging slowly
- How the learned representations are organized in embedding space

After each run, an analyzer flags problems and a tuner proposes config changes. The next run starts from those suggestions automatically.

---

## Architecture

```
model.py       Configurable 2-layer CNN (CannulationCNN)
hooks.py       Attaches forward/backward hooks to every layer,
               capturing activation and gradient stats per batch
trainer.py     Training loop that streams telemetry through hooks
analyzer.py    Reads telemetry, emits findings: dead neurons,
               vanishing/exploding gradients, overfitting, slow convergence
visualizer.py  Generates 5 matplotlib plots per run saved to plots/
tuner.py       Maps findings to config adjustments for the next run
runner.py      CLI entry point — orchestrates the full pipeline
```

Each run writes a JSON record to `experiments/` and 5 plots to `plots/`. The feedback loop is just `--iterate`: pick up the last run's suggested config and go again.

---

## Quickstart

```bash
# Install dependencies
uv sync

# First run (5 epochs, default config)
uv run cannulation

# Continue from last run's auto-tuned config
uv run cannulation --iterate

# Chain 3 self-improving runs back to back
uv run cannulation --runs 3 --iterate

# Quick smoke test
uv run cannulation --epochs 2
```

---

## Output

**Plots** (saved to `plots/<run_id>_*.png`):

| Plot | What it shows |
|---|---|
| `training_curves` | Loss and accuracy for train/val across epochs |
| `gradient_flow` | Per-layer gradient norms — flags vanishing/exploding |
| `activation_health` | Dead neuron fraction per layer across epochs |
| `weight_distributions` | Histogram of weights per layer |
| `tsne` | t-SNE projection of learned embeddings, colored by digit class |

**Experiment records** (saved to `experiments/<run_id>.json`):

```json
{
  "run_id": "20260324_134241",
  "config": { "lr": 0.001, "dropout": 0.3, "conv_channels": [32, 64], ... },
  "metrics": { "train_acc": [...], "val_acc": [...], ... },
  "findings": [
    {
      "severity": "warning",
      "layer": "training",
      "issue": "Overfitting (train-val gap=0.062)",
      "suggestion": "Increase dropout or reduce model capacity"
    }
  ],
  "next_config": { "lr": 0.001, "dropout": 0.4, ... },
  "elapsed_seconds": 82.5
}
```

---

## Analyzer Findings

The analyzer checks for four conditions after each run:

| Finding | Trigger | Tuner Response |
|---|---|---|
| Vanishing gradient | Layer grad norm < 1e-5 | Increase learning rate |
| Exploding gradient | Layer grad norm > 10 | Decrease learning rate |
| Dead neurons | >50% zero activations in a layer | Reduce dropout |
| Overfitting | Train-val accuracy gap > 5% | Increase dropout |
| Slow convergence | Val accuracy delta < 0.001 | Increase learning rate |

---

## Problem Domain

MNIST digit classification — chosen because it's well-understood, fast to iterate, and visual inputs make interpretability intuitive. The framework is not MNIST-specific; swapping in a different dataset means replacing `get_dataloaders` in `trainer.py`.

---

## Project Structure

```
cannulation-ml/
├── cannulation/
│   ├── __init__.py
│   ├── __main__.py
│   ├── model.py
│   ├── hooks.py
│   ├── trainer.py
│   ├── analyzer.py
│   ├── visualizer.py
│   ├── tuner.py
│   └── runner.py
├── experiments/        # JSON logs, one per run
├── plots/              # PNG visualizations, five per run
├── data/               # MNIST download cache
├── pyproject.toml
└── uv.lock
```
