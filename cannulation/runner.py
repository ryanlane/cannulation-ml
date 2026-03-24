import json
import os
import time
import argparse
from datetime import datetime
from typing import Dict, Any

from .model import CannulationCNN
from .hooks import HookEngine
from .trainer import Trainer
from .analyzer import Analyzer
from .visualizer import Visualizer
from .tuner import Tuner


DEFAULT_CONFIG: Dict[str, Any] = {
    "conv_channels": [32, 64],
    "fc_size": 128,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 5,
}

EXPERIMENTS_DIR = "experiments"
PLOTS_DIR = "plots"


def run_experiment(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Run {run_id}")
    print(f"  lr={config['lr']:.6f}  dropout={config['dropout']:.2f}  "
          f"conv={config['conv_channels']}  fc={config['fc_size']}")
    print(f"{'='*60}")

    model = CannulationCNN(
        conv_channels=tuple(config["conv_channels"]),
        fc_size=config["fc_size"],
        dropout=config["dropout"],
    )
    hooks = HookEngine(model)
    trainer = Trainer(model, config, hooks)

    t0 = time.time()
    metrics = trainer.train(config["epochs"])
    elapsed = time.time() - t0

    # Analyze
    analyzer = Analyzer()
    final_telemetry = metrics["epoch_telemetry"][-1]
    findings = analyzer.analyze(metrics, final_telemetry)

    if findings:
        print("\n  Analyzer findings:")
        for f in findings:
            print(f"    {f}")
    else:
        print("\n  Analyzer: no issues detected.")

    # Suggest next config
    tuner = Tuner(EXPERIMENTS_DIR)
    next_config = tuner.suggest(findings, config)

    # Persist experiment record
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    record = {
        "run_id": run_id,
        "config": config,
        "metrics": {k: v for k, v in metrics.items() if k != "epoch_telemetry"},
        "findings": [f.to_dict() for f in findings],
        "next_config": next_config,
        "elapsed_seconds": round(elapsed, 2),
    }
    record_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}.json")
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2)

    # Extract embeddings once — used for static PNG and saved for interactive viz
    print("\n  Extracting embeddings...")
    raw_emb, raw_labels = trainer.get_embeddings()
    emb_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_embeddings.json")
    with open(emb_path, "w") as fh:
        json.dump({"embeddings": raw_emb.tolist(), "labels": raw_labels.tolist()}, fh)

    # Visualize
    print("  Generating plots...")
    viz = Visualizer(PLOTS_DIR)
    plot_paths = viz.plot_all(run_id, metrics, trainer.model, raw_emb, raw_labels)
    print(f"  Saved: {', '.join(os.path.basename(p) for p in plot_paths)}")

    hooks.remove()

    final_acc = metrics["val_acc"][-1]
    print(f"\n  Final val_acc: {final_acc:.4f} | Time: {elapsed:.1f}s")
    if next_config != config:
        changes = {k: v for k, v in next_config.items() if v != config.get(k)}
        print(f"  Tuner suggests: {changes}")

    return record, next_config


def main():
    parser = argparse.ArgumentParser(description="Cannulation ML — peer inside the learning process")
    parser.add_argument("--iterate", action="store_true",
                        help="Start from last run's tuned config")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many sequential experiments to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs per run")
    args = parser.parse_args()

    tuner = Tuner(EXPERIMENTS_DIR)
    history = tuner.load_history()

    if args.iterate and history:
        config = dict(history[-1]["next_config"])
        print(f"Resuming from run {history[-1]['run_id']} with tuned config.")
    else:
        config = dict(DEFAULT_CONFIG)

    if args.epochs is not None:
        config["epochs"] = args.epochs

    for i in range(args.runs):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _, config = run_experiment(config, run_id)
        if i < args.runs - 1:
            time.sleep(1)  # ensure unique run_id timestamps


if __name__ == "__main__":
    main()
