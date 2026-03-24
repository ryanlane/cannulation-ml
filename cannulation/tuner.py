import json
import os
import re
from typing import Dict, List, Any

from .analyzer import Finding


class Tuner:
    """
    Reads analyzer findings and proposes config adjustments for the next run.
    Also persists and loads experiment history.
    """

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir

    def suggest(self, findings: List[Finding], config: Dict[str, Any]) -> Dict[str, Any]:
        next_cfg = dict(config)

        for f in findings:
            issue = f.issue
            if "Vanishing gradient" in issue:
                next_cfg["lr"] = min(next_cfg["lr"] * 2.0, 0.01)
            elif "Exploding gradient" in issue:
                next_cfg["lr"] = max(next_cfg["lr"] * 0.5, 1e-6)
            elif "Dead neurons" in issue:
                next_cfg["dropout"] = max(round(next_cfg["dropout"] - 0.1, 2), 0.0)
            elif "Overfitting" in issue:
                next_cfg["dropout"] = min(round(next_cfg["dropout"] + 0.1, 2), 0.7)
            elif "Slow convergence" in issue:
                next_cfg["lr"] = min(next_cfg["lr"] * 1.3, 0.01)

        return next_cfg

    def load_history(self) -> List[Dict]:
        if not os.path.exists(self.experiments_dir):
            return []
        runs = []
        for fname in sorted(os.listdir(self.experiments_dir)):
            if re.fullmatch(r"\d{8}_\d{6}\.json", fname):
                with open(os.path.join(self.experiments_dir, fname)) as fh:
                    runs.append(json.load(fh))
        return runs
