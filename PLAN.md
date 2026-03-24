# Cannulation ML — Feature Roadmap

## Guiding principle
Every feature should serve one of three goals:
- **See inside** — more interpretability of what the model is actually doing
- **Go faster** — less compute to reach the same result
- **Work on real tasks** — not just MNIST

---

## Phase 1 — Foundation (unblocks everything else)

### 1.1 Dataset Flexibility

**Goal:** Point the tool at your own data instead of hardcoding MNIST.

**Formats to support:**
- CSV/tabular — drop in a `.csv`, pick the target column, auto train/val split
- Image folder — directory structured as `class_name/image.jpg`
- HuggingFace datasets — one-line access via dataset name string

**Files to change:**
- `cannulation/datasets.py` — new module, `DatasetLoader` class with `.load(source, target_col, val_split)` returning standard DataLoaders
- `cannulation/trainer.py` — accept a DataLoader pair instead of calling `get_dataloaders()` internally
- `cannulation/runner.py` — accept `--data` CLI arg (path to CSV or image folder, or HF dataset name)
- `cannulation/web.py` — file upload endpoint + dataset config in the launch form
- `cannulation/templates/index.html` — dataset source selector in the run form

**Notes:**
- Tabular data needs automatic normalization and one-hot encoding of categoricals
- Image folder needs configurable resize/augmentation
- Task type (classification vs regression) should be auto-detected from the target column or configurable

---

### 1.2 Early Stopping

**Goal:** Stop training automatically when the model stops improving. Directly reduces wasted compute.

**Behavior:**
- Monitor `val_loss` by default (configurable)
- Stop if no improvement for N epochs (configurable patience, default 5)
- Restore best weights when stopping
- Log why training stopped

**Files to change:**
- `cannulation/trainer.py` — add `EarlyStopping` class, integrate into train loop
- `cannulation/runner.py` — add `--patience` CLI arg
- `cannulation/web.py` — expose patience in the run config API
- `cannulation/templates/index.html` — patience field in run form

---

### 1.3 Checkpointing

**Goal:** Save model state so runs can be resumed and best models kept.

**Behavior:**
- Save checkpoint after each epoch if val_loss improved
- Save final checkpoint at end of run
- `--resume <run_id>` loads the checkpoint from that run and continues

**Files to change:**
- `cannulation/trainer.py` — `save_checkpoint()` and `load_checkpoint()` methods
- `cannulation/runner.py` — `--resume` CLI arg
- `experiments/<run_id>/` — restructure to a directory (checkpoint + JSON + plots per run)

---

## Phase 2 — Smarter Tuning

### 2.1 Learning Rate Range Test

**Goal:** Find the best learning rate in one sweep instead of guessing across multiple runs.

**How it works:**
- Run one epoch, increasing LR exponentially from `1e-7` to `1` over all batches
- Plot loss vs. LR
- Report the LR just before loss diverges as the recommended starting point
- Write suggestion into the next run config automatically

**Files to change:**
- `cannulation/lr_finder.py` — new module, `LRFinder` class
- `cannulation/runner.py` — `--find-lr` flag runs the sweep and exits, prints recommendation
- `cannulation/web.py` — "Find LR" button on run form that runs the sweep and pre-fills the LR field
- `cannulation/visualizer.py` — `plot_lr_finder()` method
- `cannulation/templates/index.html` — "Find LR" button wired to new API endpoint

---

### 2.2 LR Scheduling

**Goal:** Let the learning rate adapt during training rather than staying fixed.

**Schedules to support:**
- `cosine` — smooth decay to near-zero
- `reduce_on_plateau` — halve LR when val_loss stalls (N epochs)
- `warmup_cosine` — linear warmup then cosine decay

**Files to change:**
- `cannulation/trainer.py` — accept `schedule` config key, attach scheduler to optimizer
- `cannulation/runner.py` / `web.py` / form — expose `schedule` as a dropdown option
- `cannulation/visualizer.py` — add LR-over-time line to training curves plot

---

### 2.3 Optuna Hyperparameter Search

**Goal:** Replace the rule-based tuner with Bayesian optimization across multiple runs.

**How it works:**
- Define a search space (LR range, dropout range, conv channel options, fc size options)
- Optuna runs N trials, each a full training run, optimizing for val_acc
- Results stored as normal experiment records so they appear in the dashboard
- Best config surfaced as the new default

**Files to change:**
- `cannulation/search.py` — new module, `HyperSearch` class wrapping Optuna
- `cannulation/runner.py` — `--search N` flag runs N Optuna trials
- `cannulation/web.py` — "Run Search" option in launch form
- `cannulation/templates/index.html` — search mode UI with trial count input
- `cannulation/templates/search.html` — new page showing trial results and best config

**Dependency:** Needs Phase 1.1 (dataset flexibility) to be useful on real tasks.

---

### 2.4 Auto-apply Analyzer Fixes

**Goal:** When the analyzer flags a problem, fix it automatically in the same run rather than waiting for the next one.

**Behaviors:**
- Exploding gradient detected mid-training → apply gradient clipping immediately
- Val loss plateau detected → trigger LR reduction (if reduce_on_plateau not already active)
- Dead neuron fraction climbing → log warning and flag for next-run architecture change

**Files to change:**
- `cannulation/trainer.py` — expose mid-training intervention hooks
- `cannulation/analyzer.py` — add `analyze_batch()` for in-flight analysis (currently only end-of-epoch)
- `cannulation/tuner.py` — add `intervene()` method called from trainer mid-run

---

## Phase 3 — Efficiency Metrics

### 3.1 FLOPs and Parameter Counting

**Goal:** Measure how much compute an architecture actually costs so efficiency can be tracked across runs.

**Metrics to track per run:**
- Total trainable parameters
- FLOPs per forward pass (single sample)
- Model size on disk (MB)
- Val accuracy per million parameters
- Val accuracy per billion FLOPs

**Files to change:**
- `cannulation/efficiency.py` — new module, `measure_model()` returning the above metrics
- `cannulation/runner.py` — call `measure_model()` before training, include in experiment record
- `cannulation/web.py` / `templates/` — show efficiency metrics on run detail page and in runs table

**Library:** `torchinfo` (lightweight, no extra deps needed for basic counts)

---

### 3.2 Pruning

**Goal:** Remove unnecessary weights after training to make the model smaller and faster.

**Two modes:**
- `magnitude` — zero out weights below a threshold, report sparsity vs. accuracy tradeoff
- `structured` — remove entire channels with lowest L1 norm, actually reduces inference FLOPs

**Workflow:**
1. Train normally
2. Prune to target sparsity (e.g. 50%)
3. Fine-tune for a few epochs
4. Record accuracy/size/FLOPs before and after

**Files to change:**
- `cannulation/pruning.py` — new module, `prune_model()` using `torch.nn.utils.prune`
- `cannulation/runner.py` — `--prune 0.5` flag triggers post-training pruning + fine-tune
- `cannulation/web.py` — pruning options in run form
- `cannulation/visualizer.py` — `plot_pruning_tradeoff()` showing sparsity vs. accuracy curve

---

### 3.3 Mixed Precision Training

**Goal:** ~2x throughput on CUDA hardware with no accuracy loss using fp16.

**Files to change:**
- `cannulation/trainer.py` — wrap forward/backward with `torch.cuda.amp.autocast()` and `GradScaler`
- `cannulation/runner.py` / `web.py` / form — `mixed_precision` boolean config option
- `cannulation/efficiency.py` — track peak GPU memory usage with and without

---

## Phase 4 — Interpretability (the core mission)

### 4.1 Grad-CAM

**Goal:** Visualize which regions of an input image most influenced the prediction.

**How it works:**
- Hook into the last conv layer's gradients and activations
- Weight activation maps by gradient magnitude
- Overlay heatmap on input image

**Files to change:**
- `cannulation/gradcam.py` — new module, `GradCAM` class
- `cannulation/visualizer.py` — `plot_gradcam_grid()` showing 10 sample images with heatmaps
- `cannulation/runner.py` — generate Grad-CAM plots at end of run if conv layers present
- `cannulation/templates/run.html` — Grad-CAM plot section

**Scope:** Only applies to image classification tasks with conv layers.

---

### 4.2 Dead Neuron Map

**Goal:** Show *which* neurons are dead (not just the fraction), so architecture problems can be localized.

**Visualization:**
- Grid heatmap per layer: each cell = one neuron, color = activation frequency
- Neurons that never fire across the validation set highlighted in red
- Trend across epochs shown as an animation or small multiples

**Files to change:**
- `cannulation/visualizer.py` — `plot_dead_neuron_map()` method
- `cannulation/hooks.py` — track per-neuron activation frequency (not just mean fraction)
- `cannulation/templates/run.html` — dead neuron map section

---

### 4.3 Loss Landscape

**Goal:** Visualize the shape of the loss surface around the trained weights — flat minima generalize better than sharp ones.

**How it works:**
- Sample two random direction vectors in weight space
- Evaluate loss at a grid of points around the current weights
- Plot as a 2D surface or contour map

**Files to change:**
- `cannulation/landscape.py` — new module, `LossLandscape` class
- `cannulation/visualizer.py` — `plot_loss_landscape()` method
- `cannulation/runner.py` — `--landscape` flag (expensive, opt-in)
- `cannulation/templates/run.html` — landscape plot section

**Note:** This is compute-intensive (N×N forward passes). Default grid should be small (20×20). Make it opt-in.

---

## Phase 5 — Run Comparison

### 5.1 Overlay Charts

**Goal:** Compare multiple runs on a single chart to see the effect of config changes.

**Charts:**
- Loss curves for N selected runs overlaid, color-coded
- Val accuracy vs. epoch for N runs
- Accuracy vs. FLOPs scatter (once Phase 3.1 is done)

**Files to change:**
- `cannulation/web.py` — `GET /compare?runs=id1,id2,id3` route
- `cannulation/templates/compare.html` — new page with overlaid charts (Chart.js, no extra Python deps)
- `cannulation/templates/index.html` — checkboxes on run rows + "Compare selected" button

---

### 5.2 Config Diff

**Goal:** Show exactly what changed between two runs and what effect it had.

**Files to change:**
- `cannulation/web.py` — include config diff data in the compare route
- `cannulation/templates/compare.html` — config diff table: key / run A value / run B value / Δ val_acc

---

## Dependency order

```
1.1 Dataset Flexibility  ←  everything real depends on this
1.2 Early Stopping       ←  standalone, do early
1.3 Checkpointing        ←  standalone, do early
2.1 LR Range Test        ←  requires 1.1 to be meaningful
2.2 LR Scheduling        ←  requires trainer changes from 1.2
2.3 Optuna Search        ←  requires 1.1, benefits from 2.1 + 2.2
2.4 Auto-apply Fixes     ←  requires 2.2
3.1 FLOPs / Params       ←  standalone
3.2 Pruning              ←  requires 3.1
3.3 Mixed Precision      ←  standalone
4.1 Grad-CAM             ←  requires 1.1 (image tasks)
4.2 Dead Neuron Map      ←  requires hooks.py changes
4.3 Loss Landscape       ←  standalone, opt-in
5.1 Overlay Charts       ←  requires 3.1 for full value
5.2 Config Diff          ←  requires 5.1
```

## New files summary

| File | Purpose |
|---|---|
| `cannulation/datasets.py` | Universal dataset loader (CSV, image folder, HuggingFace) |
| `cannulation/lr_finder.py` | LR range test sweep |
| `cannulation/search.py` | Optuna hyperparameter search |
| `cannulation/efficiency.py` | FLOPs, parameter count, memory tracking |
| `cannulation/pruning.py` | Magnitude and structured pruning |
| `cannulation/gradcam.py` | Grad-CAM implementation |
| `cannulation/landscape.py` | Loss landscape visualization |
| `cannulation/templates/compare.html` | Multi-run comparison page |
| `cannulation/templates/search.html` | Optuna search results page |
