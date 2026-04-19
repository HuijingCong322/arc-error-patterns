import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

ARC_DIR   = ROOT / "ARC-AGI" / "data" / "evaluation"
VARC_DIR  = ROOT / "VARC_predictions"
HARC_DIR  = ROOT / "HARC" / "data"


# ── ARC ground truth ──────────────────────────────────────────────────────────

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


# ── Grid parsing ──────────────────────────────────────────────────────────────

def parse_grid(value):
    """
    Parse H-ARC pipe-encoded grid string into a 2-D int list.

    Format:  |012|345|678|
    Returns: [[0,1,2],[3,4,5],[6,7,8]]
    Also handles JSON-list strings and plain list objects.
    """
    if value is None or (isinstance(value, float)):
        return None
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value.startswith("|"):
        rows = [r for r in value.split("|") if r]
        return [[int(c) for c in row] for row in rows]
    return json.loads(value)


# ── H-ARC human responses ─────────────────────────────────────────────────────

def load_harc_summary():
    """
    Load summary_data.csv — one row per participant attempt.

    Columns used downstream:
      task_name       e.g. "6e19193c.json"  → strip ".json" for task_id
      hashed_id       participant identifier
      attempt_number  1 / 2 / 3
      solved          "true" / "false"
      test_output_grid  pipe-encoded grid string
      task_type       filter to "evaluation" for the 400-task set
    """
    path = HARC_DIR / "summary_data.csv"
    df = pd.read_csv(path)
    df = df[df["task_type"] == "evaluation"].copy()
    df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)
    print(f"H-ARC summary_data: {len(df)} rows, {df['task_id'].nunique()} tasks, "
          f"{df['hashed_id'].nunique()} participants")
    return df


def load_harc_incorrect_submissions():
    """
    Load incorrect_submissions.csv — aggregated wrong answers per task.

    Columns:
      task_name, task_type, test_output_grid (pipe-encoded), count
    Returns only evaluation tasks.
    """
    path = HARC_DIR / "incorrect_submissions.csv"
    df = pd.read_csv(path)
    df = df[df["task_type"] == "evaluation"].copy()
    df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)
    return df
