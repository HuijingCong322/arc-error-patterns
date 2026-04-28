"""
Human answer-distribution metrics — the human-side counterpart to
view_consistency.py.

For each evaluation task, treat each participant's LAST attempt as one
"sample" of how a human approaches the task, and compute the distribution
over distinct answer grids. The resulting entropy / top-1 share is directly
comparable to the VARC view-distribution metrics.

Per-task metrics
----------------
human_n_responses        int    - participant count (≈ 10 per task)
human_n_unique_grids     int    - number of distinct final-answer grids
human_top1_share         float  - top-1 grid count / n_responses
human_top1_correct       bool   - is the most-common human answer the correct one
human_top1_top2_gap      float  - confidence margin between human top-1 and top-2
human_vote_entropy       float  - Shannon entropy (nats) over participant answers
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.load_data import load_arc_ground_truth, load_harc_summary, parse_grid


def _vote_counts(grids):
    counts = {}
    grid_map = {}
    for g in grids:
        if g is None:
            continue
        key = json.dumps(g)
        grid_map[key] = g
        counts[key] = counts.get(key, 0) + 1
    return counts, grid_map


def _entropy(counts, total):
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def _grids_match(a, b):
    if len(a) != len(b):
        return False
    return all(list(r1) == list(r2) for r1, r2 in zip(a, b))


def compute_human_consistency(harc_df=None, ground_truth=None):
    """
    Returns DataFrame with one row per task_id and human-side distribution metrics.
    """
    if harc_df is None:
        harc_df = load_harc_summary()
    if ground_truth is None:
        ground_truth = load_arc_ground_truth()

    last = (
        harc_df.sort_values("attempt_number")
        .groupby(["task_id", "hashed_id"], sort=False)
        .last()
        .reset_index()
    )

    rows = []
    for task_id, group in last.groupby("task_id"):
        truths = ground_truth.get(task_id)
        if truths is None:
            continue
        truth = truths[0]   # evaluation tasks have one test example

        grids = [parse_grid(g) for g in group["test_output_grid"]]
        grids = [g for g in grids if g is not None]
        n = len(grids)
        if n == 0:
            continue

        counts, grid_map = _vote_counts(grids)
        sorted_keys = sorted(counts, key=counts.get, reverse=True)
        top1 = counts[sorted_keys[0]]
        top2 = counts[sorted_keys[1]] if len(sorted_keys) > 1 else 0
        top1_grid = grid_map[sorted_keys[0]]

        rows.append({
            "task_id":              task_id,
            "human_n_responses":    n,
            "human_n_unique_grids": len(counts),
            "human_top1_share":     top1 / n,
            "human_top1_correct":   _grids_match(top1_grid, truth),
            "human_top1_top2_gap":  (top1 - top2) / n,
            "human_vote_entropy":   _entropy(counts, n),
        })

    df = pd.DataFrame(rows)
    print(f"Human consistency computed: {len(df)} tasks")
    return df


if __name__ == "__main__":
    df = compute_human_consistency()
    out = ROOT / "data" / "human_consistency.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
