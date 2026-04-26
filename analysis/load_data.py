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

def _vote_entries(grids):
    counts = {}
    grid_map = {}
    for grid in grids:
        key = json.dumps(grid)
        grid_map[key] = grid
        counts[key] = counts.get(key, 0) + 1
    sorted_votes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"prediction": grid_map[key], "votes": votes} for key, votes in sorted_votes]


def _get_varc_attempt_dir(model, attempt):
    attempt_dir = VARC_DIR / "VARC_predictions" / model / f"attempt_{attempt}"
    if not attempt_dir.exists():
        raise FileNotFoundError(
            f"Attempt directory not found: {attempt_dir}\n"
            f"Available: {[p.name for p in (VARC_DIR / 'VARC_predictions' / model).iterdir()]}"
        )
    return attempt_dir


def load_varc_vote_entries(model="ARC-1_ViT", attempt=0, top_k=None):
    """
    Official plain VARC loading: use a single attempt directory and majority-vote
    within that run's 510 multi-view predictions for each test example.

    Returns:
      dict[task_id -> dict[test_idx -> list[{prediction, votes}]]]
    """
    attempt_dir = _get_varc_attempt_dir(model, attempt)
    result = {}
    for path in sorted(attempt_dir.glob("*_predictions.json")):
        task_id = path.stem.replace("_predictions", "")
        with open(path) as f:
            raw = json.load(f)      # {"0": [grid, ...], "1": [...], ...}
        result[task_id] = {}
        for k, grids in raw.items():
            entries = _vote_entries(grids)
            if top_k is not None:
                entries = entries[:top_k]
            result[task_id][int(k)] = entries
    print(
        f"VARC vote entries loaded: {len(result)} tasks "
        f"(model={model}, attempt={attempt}, top_k={top_k})"
    )
    return result


def load_varc_predictions(model="ARC-1_ViT", attempt=0):
    """
    dict[task_id -> list[grid]]

    Aligns with official plain VARC analysis:
      - use a single attempt directory (official repo uses attempt_0 for ViT)
      - for each test example, majority-vote within that attempt's 510 views
      - return the top-1 prediction grid for downstream visualization/error typing
    """
    vote_entries = load_varc_vote_entries(model=model, attempt=attempt, top_k=1)
    result = {}
    for task_id, by_idx in vote_entries.items():
        result[task_id] = [by_idx[i][0]["prediction"] for i in range(len(by_idx))]
    print(f"VARC predictions loaded: {len(result)} tasks (model={model}, attempt={attempt})")
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
