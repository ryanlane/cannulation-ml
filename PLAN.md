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

### 4.4 Training-Aware Graph Inspector

**Goal:** Borrow the best part of tools like Netron: a navigable model graph with shapes, params, operator attributes, and layer-level detail panels. But make it training-aware by overlaying runtime telemetry on top of the static graph.

**What to show per node:**
- Layer type and module path
- Input/output shapes
- Trainable parameter count
- Estimated FLOPs / MACs
- Operator attributes (kernel size, stride, padding, activation type, etc.)
- Attached runtime badges: vanishing gradients, exploding gradients, dead neurons, saturation, over-compression

**UI behaviors:**
- Search by layer name or type
- Filter to only problematic layers
- Expand a node to see static properties + runtime stats + related charts
- Color graph edges/nodes by health score

**Files to change:**
- `cannulation/graph.py` — new module, normalized internal graph model (`ModelGraph`, `GraphNode`, `GraphEdge`, `TensorSpec`)
- `cannulation/model_inspector.py` — new module, extracts shapes, params, and operator metadata from the PyTorch model
- `cannulation/runner.py` — save graph metadata into the experiment record or sibling JSON file
- `cannulation/web.py` — route/API for graph payloads on the run detail page
- `cannulation/templates/run.html` — graph explorer section with node detail drawer

**Borrow from Netron:**
- Graph / node / tensor abstraction
- Shape and dtype as first-class metadata
- Node detail panels driven by operator metadata

---

### 4.5 Rich Runtime Telemetry

**Goal:** Replace thin mean/std summaries with distributions and optimizer-aware diagnostics that explain how the model is actually learning.

**Add per layer:**
- Activation percentiles (`p1`, `p5`, `p50`, `p95`, `p99`)
- Positive/negative fraction and saturation fraction
- Channel-wise activation summaries for conv layers
- Gradient percentiles and near-zero gradient fraction
- Gradient-to-weight norm ratio
- Update-to-weight ratio after optimizer step
- Telemetry drift across epochs, not just final snapshots

**Sampling strategy:**
- Keep epoch summaries for cheap storage
- Also retain sampled batch-level telemetry for the first, middle, and last portion of each epoch
- Persist enough raw distribution data to render boxplots, histograms, and trend charts later

**Files to change:**
- `cannulation/hooks.py` — capture richer activation/gradient stats and per-channel summaries
- `cannulation/trainer.py` — collect sampled batch telemetry, optimizer/update ratios, and LR-over-time
- `cannulation/analyzer.py` — promote from threshold-only checks to distribution-aware analysis
- `cannulation/visualizer.py` — new plots for distributions, drift, and update ratios
- `cannulation/templates/run.html` — telemetry drill-down panels per layer

---

### 4.6 Error and Slice Explorer

**Goal:** Improve the model by understanding what it gets wrong, not just whether val accuracy went up.

**What to track per run:**
- Confusion matrix
- Per-class precision / recall / F1
- Highest-loss examples
- Most uncertain predictions
- Hardest class pairs
- Calibration metrics (ECE + reliability diagram)
- Slice metrics by class, source, metadata bucket, or feature range

**Why this matters:**
- Many failures are data problems or slice-specific generalization failures, not architecture problems
- This exposes whether the next action should be more data, rebalancing, augmentation, relabeling, or architecture changes

**Files to change:**
- `cannulation/evaluation.py` — new module, computes confusion matrices, per-class metrics, calibration, and hardest examples
- `cannulation/datasets.py` — retain slice metadata / sample identifiers for later analysis
- `cannulation/runner.py` — save evaluation payloads alongside run metrics
- `cannulation/web.py` — API endpoints for confusion matrix and slice explorer data
- `cannulation/templates/run.html` — error analysis section with confusion heatmap and slice tables

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

### 5.3 Model and Telemetry Diff

**Goal:** Compare two runs structurally, not just by scalar metrics, so you can see which layers changed and whether the change actually improved training behavior.

**Diff views:**
- Layer-by-layer static graph diff (shapes, params, FLOPs, operator attributes)
- Layer-by-layer telemetry delta (activation health, gradient health, update ratios)
- Per-class metric delta and confusion matrix delta
- Embedding separability delta and hardest-class-pair changes

**Files to change:**
- `cannulation/web.py` — extend compare route with graph + telemetry diff payloads
- `cannulation/templates/compare.html` — side-by-side model inspector and telemetry diff tables
- `cannulation/analyzer.py` — summarize the most important run-to-run regressions and improvements

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
4.4 Graph Inspector      ←  benefits from 3.1 FLOPs / params
4.5 Rich Telemetry       ←  requires hooks.py + trainer.py changes
4.6 Error / Slice Explorer ← requires 1.1 dataset flexibility
5.1 Overlay Charts       ←  requires 3.1 for full value
5.2 Config Diff          ←  requires 5.1
5.3 Model / Telemetry Diff ← requires 4.4 + 4.5 + 5.1
```

## Recommended implementation sequence

The roadmap above is the full feature set. This is the practical build order for the current codebase if the goal is to improve the model fastest while keeping the implementation coherent.

### Milestone A — Make runs worth comparing

**Why first:** The current system still lacks the basic data needed to explain why a run improved or regressed. Before adding more UI, make each run produce better artifacts.

**Build in this order:**
1. `1.1 Dataset Flexibility`
2. `3.1 FLOPs and Parameter Counting`
3. `4.6 Error and Slice Explorer` (metrics payload only first, UI second)

**Concrete repo work:**
- `cannulation/datasets.py` — retain sample identifiers and slice metadata
- `cannulation/trainer.py` — expose predictions, losses, confidences, and labels from validation
- `cannulation/runner.py` — save efficiency + evaluation payloads into experiment artifacts
- `cannulation/web.py` — surface these metrics on the run detail page and run list

**Definition of done:**
- Every run records per-class metrics, confusion data, hardest examples, and efficiency stats
- Two runs can already be compared meaningfully even before graph inspection exists

---

### Milestone B — Upgrade telemetry from summary stats to diagnostics

**Why second:** The analyzer and charts are bottlenecked by weak hook data. Richer telemetry unlocks better charts, better alerts, and later graph overlays.

**Build in this order:**
1. `4.5 Rich Runtime Telemetry`
2. `2.2 LR Scheduling` with LR-over-time capture
3. `2.4 Auto-apply Analyzer Fixes` after telemetry is trustworthy

**Concrete repo work:**
- `cannulation/hooks.py` — percentiles, saturation, channel summaries, gradient sparsity
- `cannulation/trainer.py` — sampled batch telemetry, update-to-weight ratio, LR trace, optimizer-step stats
- `cannulation/analyzer.py` — switch from threshold-only logic to distribution-aware findings
- `cannulation/visualizer.py` — distribution charts, drift charts, and optimizer-health charts

**Definition of done:**
- A run page can explain optimization behavior layer by layer, not just show a final warning badge
- The analyzer starts producing fewer generic suggestions and more layer-specific ones

---

### Milestone C — Add the training-aware graph inspector

**Why third:** Once runs have strong static and runtime data, the graph view becomes a force multiplier instead of a pretty shell.

**Build in this order:**
1. `4.4 Training-Aware Graph Inspector`
2. Integrate `3.1` FLOPs/params into graph nodes
3. Overlay `4.5` telemetry onto nodes and edges

**Concrete repo work:**
- `cannulation/graph.py` — stable internal graph schema
- `cannulation/model_inspector.py` — PyTorch graph extraction, shapes, params, operator metadata
- `cannulation/runner.py` — persist graph metadata per run
- `cannulation/web.py` — graph API + node detail payloads
- `cannulation/templates/run.html` — graph explorer with search/filter/detail panel

**Definition of done:**
- Clicking a layer shows its shape, params, operator attributes, activation health, gradient health, and related charts
- The graph view is useful enough to debug bad architectures without reading raw model code

---

### Milestone D — Make comparison a first-class workflow

**Why fourth:** Improvement work is comparative. Once single-run inspection is strong, the next leverage point is showing what changed between runs.

**Build in this order:**
1. `5.1 Overlay Charts`
2. `5.2 Config Diff`
3. `5.3 Model and Telemetry Diff`

**Concrete repo work:**
- `cannulation/web.py` — comparison routes returning config, metric, graph, and telemetry deltas
- `cannulation/templates/compare.html` — overlay charts + diff tables + side-by-side inspector views
- `cannulation/analyzer.py` — summarize the most important regression or improvement between runs

**Definition of done:**
- You can answer "what changed?" and "did it help?" from the compare page alone

---

### Milestone E — Add expensive interpretability and optimization extras

**Why last:** These are valuable, but they should come after the observability foundation. Otherwise they produce isolated visualizations without enough context.

**Build in this order:**
1. `4.1 Grad-CAM`
2. `4.2 Dead Neuron Map`
3. `4.3 Loss Landscape`
4. `3.2 Pruning`
5. `2.3 Optuna Search`

**Concrete repo work:**
- Add these only after run artifacts, graph metadata, and comparison workflows already exist
- Make all expensive features opt-in and persist their outputs as reusable run artifacts

**Definition of done:**
- These features augment decision-making rather than becoming disconnected demo pages

---

## If we want the single best next step

Start with a narrow vertical slice of `4.5 Rich Runtime Telemetry` plus the data payload half of `4.6 Error and Slice Explorer`.

That combination gives the fastest payoff because it improves:
- The analyzer
- The charts
- The eventual graph inspector
- Run comparison
- Model tuning decisions

**Suggested first PR sequence:**
1. Expand `cannulation/hooks.py` telemetry schema
2. Expand `cannulation/trainer.py` validation outputs and optimizer diagnostics
3. Add `cannulation/evaluation.py` and save confusion/per-class metrics in `cannulation/runner.py`
4. Update `cannulation/web.py` and `cannulation/templates/run.html` to show the new diagnostics

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
| `cannulation/graph.py` | Internal graph representation for model inspection |
| `cannulation/model_inspector.py` | Extract shapes, params, FLOPs, and operator metadata from PyTorch models |
| `cannulation/evaluation.py` | Confusion matrices, per-class metrics, calibration, and hardest-example analysis |
| `cannulation/templates/compare.html` | Multi-run comparison page |
| `cannulation/templates/search.html` | Optuna search results page |
