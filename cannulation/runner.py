import json
import os
import time
import argparse
from datetime import datetime
from typing import Dict, Any

from .datasets import DatasetLoader
from .hooks import HookEngine
from .model import build_model
from .trainer import Trainer
from .analyzer import Analyzer
from .embedding_metrics import compute_metrics
from .visualizer import Visualizer
from .tuner import Tuner


DEFAULT_CONFIG: Dict[str, Any] = {
    "conv_channels": [32, 64],
    "fc_size": 128,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 5,
    "data_source": None,
    "target_col": None,
    "val_split": 0.2,
}

EXPERIMENTS_DIR = "experiments"


def run_experiment(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Run {run_id}")
    print(f"  lr={config['lr']:.6f}  dropout={config['dropout']:.2f}  "
          f"conv={config['conv_channels']}  fc={config['fc_size']}")
    print(f"{'='*60}")

    dataset_loader = DatasetLoader()
    train_loader, val_loader, dataset_info = dataset_loader.load(
        source=config.get("data_source"),
        target_col=config.get("target_col"),
        val_split=config.get("val_split", 0.2),
        batch_size=config["batch_size"],
    )
    print(f"  data={dataset_info.source}  type={dataset_info.dataset_type}  task={dataset_info.task_type}")

    model = build_model(config, dataset_info)
    hooks = HookEngine(model)
    trainer = Trainer(model, config, hooks, train_loader, val_loader, dataset_info)

    t0 = time.time()
    metrics = trainer.train(config["epochs"])
    elapsed = time.time() - t0

    analyzer = Analyzer()
    tuner = Tuner(EXPERIMENTS_DIR)
    final_telemetry = metrics["epoch_telemetry"][-1]
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Extract embeddings before analysis so metrics can inform findings
    raw_emb, raw_labels = trainer.get_embeddings()
    emb_metrics = None
    if raw_emb is not None and raw_labels is not None and len(set(raw_labels.tolist())) > 1:
        print("\n  Extracting embeddings...")
        emb_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_embeddings.json")
        with open(emb_path, "w") as fh:
            json.dump({"embeddings": raw_emb.tolist(), "labels": raw_labels.tolist()}, fh)

        print("  Computing embedding metrics...")
        try:
            emb_metrics = compute_metrics(raw_emb, raw_labels)
        except ValueError as exc:
            print(f"  Skipping embedding metrics: {exc}")

    findings = analyzer.analyze(metrics, final_telemetry, emb_metrics)
    next_config = tuner.suggest(findings, config)

    if findings:
        print("\n  Analyzer findings:")
        for f in findings:
            print(f"    {f}")
    else:
        print("\n  Analyzer: no issues detected.")

    record = {
        "run_id": run_id,
        "config": config,
        "dataset": dataset_info.to_dict(),
        "metrics": {k: v for k, v in metrics.items() if k != "epoch_telemetry"},
        "embedding_metrics": emb_metrics,
        "findings": [f.to_dict() for f in findings],
        "next_config": next_config,
        "elapsed_seconds": round(elapsed, 2),
    }
    record_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}.json")
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2)

    # Visualize
    print("\n  Building interactive charts...")
    viz = Visualizer(EXPERIMENTS_DIR)
    chart_path = viz.save_all(run_id, metrics, trainer.model)
    print(f"  Saved: {os.path.basename(chart_path)}")

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
    parser.add_argument("--data", dest="data_source", default=None,
                        help="Dataset source: CSV path, image folder path, HF dataset name, or 'mnist'")
    parser.add_argument("--target-col", default=None,
                        help="Target column for CSV or HuggingFace tabular datasets")
    parser.add_argument("--val-split", type=float, default=None,
                        help="Validation split fraction for non-MNIST datasets")
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
    if args.data_source is not None:
        config["data_source"] = args.data_source
    if args.target_col is not None:
        config["target_col"] = args.target_col
    if args.val_split is not None:
        config["val_split"] = args.val_split

    for i in range(args.runs):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _, config = run_experiment(config, run_id)
        if i < args.runs - 1:
            time.sleep(1)  # ensure unique run_id timestamps


if __name__ == "__main__":
    main()
