import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

ARC_DIR   = ROOT / "ARC-AGI" / "data" / "evaluation"
VARC_DIR  = ROOT / "VARC_predictions"
HARC_DIR  = ROOT / "HARC" / "data"


# ── ARC ground truth ─────────────────────────────────────────────────────────

def load_arc_ground_truth():
    """dict[task_id -> list[grid]]  (one grid per test example)"""
    result = {}
    for path in ARC_DIR.glob("*.json"):
        with open(path) as f:
            task = json.load(f)
        result[path.stem] = [ex["output"] for ex in task["test"]]
    return result


# ── VARC predictions ──────────────────────────────────────────────────────────

def _majority_vote(grids):
    counts = {}
    for g in grids:
        key = tuple(tuple(row) for row in g)
        counts[key] = counts.get(key, 0) + 1
    return [list(row) for row in max(counts, key=counts.get)]


def load_varc_predictions():
    """dict[task_id -> list[grid]]  majority-voted, ordered by test index"""
    result = {}
    for path in VARC_DIR.glob("*_predictions.json"):
        task_id = path.stem.replace("_predictions", "")
        with open(path) as f:
            raw = json.load(f)          # {"0": [grid, grid, ...], ...}
        voted = {int(k): _majority_vote(v) for k, v in raw.items()}
        result[task_id] = [voted[i] for i in range(len(voted))]
    return result


# ── H-ARC human responses ─────────────────────────────────────────────────────

def load_harc_responses():
    """
    Returns raw DataFrame from H-ARC CSVs.

    H-ARC OSF layout (https://osf.io/c73kw/):
      HARC/data/
        ├── responses.csv     ← main file: one row per attempt
        ├── tasks.csv
        └── participants.csv

    Key columns expected in responses.csv:
      task_id, participant_id, attempt (1-3), is_correct, response_grid (JSON string)

    Run `df.columns.tolist()` after loading to verify column names.
    """
    candidates = ["responses.csv", "arc_responses.csv", "results.csv", "data.csv"]
    for name in candidates:
        path = HARC_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded H-ARC: {path.name}  shape={df.shape}")
            return df

    available = [p.name for p in HARC_DIR.glob("*.csv")]
    raise FileNotFoundError(
        f"H-ARC responses CSV not found in {HARC_DIR}\n"
        f"Files present: {available}\n"
        "Update `candidates` list above with the correct filename."
    )


def parse_grid(value):
    """Parse a grid stored as JSON string or return as-is if already a list."""
    if value is None or (isinstance(value, float)):
        return None
    if isinstance(value, list):
        return value
    return json.loads(value)
