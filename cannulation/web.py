import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

PLOTS_DIR = Path("plots")
EXPERIMENTS_DIR = Path("experiments")
TEMPLATES_DIR = Path(__file__).parent / "templates"

PLOT_NAMES = [
    ("training_curves", "Training Curves"),
    ("gradient_flow", "Gradient Flow"),
    ("activation_health", "Activation Health"),
    ("weight_distributions", "Weight Distributions"),
    ("tsne", "t-SNE Embeddings"),
]

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

app = FastAPI(title="Cannulation ML")
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory job state — one active run at a time
active_runs: dict = {}


def _load_chart_bundle(run_id: str) -> list:
    path = EXPERIMENTS_DIR / f"{run_id}_charts.json"
    if not path.exists():
        return []
    with open(path) as f:
        payload = json.load(f)
    return payload.get("charts", [])


class LineLogger:
    """Redirects stdout writes into a list of log lines."""
    def __init__(self, log_list: list):
        self._log = log_list
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._log.append(line)

    def flush(self):
        if self._buf.strip():
            self._log.append(self._buf)
            self._buf = ""


class RunConfig(BaseModel):
    conv_channels: List[int] = [32, 64]
    fc_size: int = 128
    dropout: float = 0.3
    lr: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    patience: int = 0
    schedule: Optional[str] = None
    data_source: Optional[str] = None
    target_col: Optional[str] = None
    val_split: float = 0.2
    iterate: bool = False


def _load_runs() -> list:
    runs = []
    for path in sorted(EXPERIMENTS_DIR.glob("????????_??????.json"), reverse=True):
        with open(path) as f:
            data = json.load(f)
        findings = data.get("findings", [])
        severities = [f["severity"] for f in findings]
        worst = (
            "critical" if "critical" in severities
            else "warning" if "warning" in severities
            else "info" if severities
            else None
        )
        runs.append({
            "run_id": data["run_id"],
            "val_acc": data["metrics"]["val_acc"][-1],
            "train_acc": data["metrics"]["train_acc"][-1],
            "epochs": len(data["metrics"]["val_acc"]),
            "elapsed": data["elapsed_seconds"],
            "findings_count": len(findings),
            "worst_severity": worst,
            "lr": data["config"]["lr"],
            "dropout": data["config"]["dropout"],
        })
    return runs


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    runs = _load_runs()
    best_acc = max((r["val_acc"] for r in runs), default=None)

    # Surface any active run so the UI can resume polling
    active = next(
        (job for job in active_runs.values() if job["status"] == "running"),
        None,
    )

    # Default config: last tuner suggestion or hardcoded defaults
    from .tuner import Tuner
    history = Tuner(EXPERIMENTS_DIR).load_history()
    defaults = {
        "conv_channels": [32, 64], "fc_size": 128, "dropout": 0.3,
        "lr": 0.001, "batch_size": 64, "epochs": 5, "patience": 0,
        "schedule": None, "data_source": None, "target_col": None, "val_split": 0.2,
    }
    default_cfg = {**defaults, **history[-1]["next_config"]} if history else defaults

    return templates.TemplateResponse(request, "index.html", {
        "runs": runs,
        "best_acc": best_acc,
        "active_run": active,
        "default_cfg": default_cfg,
    })


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    path = EXPERIMENTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    with open(path) as f:
        data = json.load(f)

    charts = _load_chart_bundle(run_id)
    all_plots = [
        {"slug": slug, "title": title, "url": f"/plots/{run_id}_{slug}.png"}
        for slug, title in PLOT_NAMES
        if (PLOTS_DIR / f"{run_id}_{slug}.png").exists()
    ]
    legacy_plots = [p for p in all_plots if p["slug"] != "tsne"]
    static_tsne = next((p for p in all_plots if p["slug"] == "tsne"), None)
    has_embeddings = (EXPERIMENTS_DIR / f"{run_id}_embeddings.json").exists()

    epochs = list(range(1, len(data["metrics"]["val_acc"]) + 1))
    epoch_rows = list(zip(
        epochs,
        data["metrics"]["train_loss"],
        data["metrics"]["train_acc"],
        data["metrics"]["val_loss"],
        data["metrics"]["val_acc"],
    ))

    return templates.TemplateResponse(request, "run.html", {
        "run": data,
        "charts": charts,
        "legacy_plots": legacy_plots,
        "static_tsne": static_tsne,
        "has_embeddings": has_embeddings,
        "epoch_rows": epoch_rows,
        "emb_metrics": data.get("embedding_metrics"),
        "efficiency": data.get("efficiency"),
        "evaluation": data.get("evaluation"),
    })


@app.post("/api/run/start")
async def start_run(config: RunConfig):
    # Only one run at a time
    for job in active_runs.values():
        if job["status"] == "running":
            return JSONResponse(
                {"error": "A run is already in progress"},
                status_code=409,
            )

    cfg = config.model_dump(exclude={"iterate"})

    if config.iterate:
        from .tuner import Tuner
        history = Tuner(EXPERIMENTS_DIR).load_history()
        if history:
            cfg = dict(history[-1]["next_config"])
            cfg["epochs"] = config.epochs  # honour explicit epoch override
            cfg["patience"] = config.patience
            if config.data_source is not None:
                cfg["data_source"] = config.data_source
            if config.target_col is not None:
                cfg["target_col"] = config.target_col
            cfg["val_split"] = config.val_split

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = {"status": "running", "logs": [], "run_id": run_id}
    active_runs[run_id] = job

    def _run():
        logger = LineLogger(job["logs"])
        old_stdout = sys.stdout
        sys.stdout = logger
        try:
            from .runner import run_experiment
            run_experiment(cfg, run_id)
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            sys.stdout = old_stdout

    threading.Thread(target=_run, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/status")
async def run_status(run_id: str):
    if run_id not in active_runs:
        return {"status": "unknown", "run_id": run_id}
    return active_runs[run_id]


@app.get("/api/run/{run_id}/embeddings")
async def run_embeddings(run_id: str, dims: int = 3):
    if dims < 2 or dims > 5:
        raise HTTPException(status_code=400, detail="dims must be 2–5")

    cache_path = EXPERIMENTS_DIR / f"{run_id}_tsne_{dims}d.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    raw_path = EXPERIMENTS_DIR / f"{run_id}_embeddings.json"
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Embeddings not found — re-run to generate")

    import numpy as np
    from sklearn.manifold import TSNE

    with open(raw_path) as f:
        data = json.load(f)

    coords = TSNE(
        n_components=dims,
        random_state=42,
        perplexity=30,
        method="exact" if dims > 3 else "barnes_hut",
    ).fit_transform(np.array(data["embeddings"]))

    result = {"coords": coords.tolist(), "labels": data["labels"], "dims": dims}
    with open(cache_path, "w") as f:
        json.dump(result, f)

    return result


@app.get("/api/default-config")
async def default_config():
    from .tuner import Tuner
    history = Tuner(EXPERIMENTS_DIR).load_history()
    if history:
        return {
            "conv_channels": [32, 64],
            "fc_size": 128,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 5,
            "patience": 0,
            "schedule": None,
            "data_source": None,
            "target_col": None,
            "val_split": 0.2,
            **history[-1]["next_config"],
        }
    return {"conv_channels": [32, 64], "fc_size": 128, "dropout": 0.3,
            "lr": 0.001, "batch_size": 64, "epochs": 5, "patience": 0,
            "schedule": None, "data_source": None, "target_col": None, "val_split": 0.2}


def start():
    import uvicorn
    uvicorn.run("cannulation.web:app", host="0.0.0.0", port=8000, reload=True)
