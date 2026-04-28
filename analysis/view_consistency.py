"""
View-consistency metrics over VARC's 510 random-view predictions.

VARC inference samples 510 augmented views per test input and majority-votes
their predictions. The full 510-grid distribution is itself a "passive input
intervention" — we can read consistency / confidence signals out of it without
re-running the model.

Per (task_id, test_idx) metrics
-------------------------------
n_views               int    - total view-level predictions (typically 510)
n_unique_grids        int    - distinct prediction grids
view_level_accuracy   float  - fraction of views matching ground truth exactly
top1_share            float  - top-1 grid vote count / n_views
top2_share            float  - top-2 grid vote count / n_views (0 if only 1 unique)
top1_top2_gap         float  - top1_share - top2_share
vote_entropy          float  - Shannon entropy (nats) over view votes
top1_correct          bool   - is the majority-vote top-1 grid correct
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.load_data import VARC_DIR, load_arc_ground_truth
from analysis.error_analysis import exact_match


def _vote_counts(grids):
    counts = {}
    grid_map = {}
    for grid in grids:
        key = json.dumps(grid)
        grid_map[key] = grid
        counts[key] = counts.get(key, 0) + 1
    return counts, grid_map


def _entropy(counts, total):
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def compute_view_consistency(model="ARC-1_ViT", attempt=0, ground_truth=None):
    """
    Returns DataFrame indexed by (task_id, test_idx) with view-level metrics.
    Ground truth is loaded if not provided.
    """
    if ground_truth is None:
        ground_truth = load_arc_ground_truth()

    attempt_dir = VARC_DIR / "VARC_predictions" / model / f"attempt_{attempt}"
    rows = []
    for path in sorted(attempt_dir.glob("*_predictions.json")):
        task_id = path.stem.replace("_predictions", "")
        truths = ground_truth.get(task_id)
        if truths is None:
            continue
        with open(path) as f:
            raw = json.load(f)

        for k, grids in raw.items():
            test_idx = int(k)
            if test_idx >= len(truths):
                continue
            truth = truths[test_idx]
            n_views = len(grids)

            counts, grid_map = _vote_counts(grids)
            sorted_counts = sorted(counts.values(), reverse=True)
            top1 = sorted_counts[0]
            top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0

            top1_key = max(counts, key=counts.get)
            top1_grid = grid_map[top1_key]

            view_correct = sum(1 for g in grids if exact_match(g, truth))

            rows.append({
                "task_id":             task_id,
                "test_idx":            test_idx,
                "n_views":             n_views,
                "n_unique_grids":      len(counts),
                "view_level_accuracy": view_correct / n_views if n_views else 0.0,
                "top1_share":          top1 / n_views if n_views else 0.0,
                "top2_share":          top2 / n_views if n_views else 0.0,
                "top1_top2_gap":       (top1 - top2) / n_views if n_views else 0.0,
                "vote_entropy":        _entropy(counts, n_views),
                "top1_correct":        exact_match(top1_grid, truth),
            })

    df = pd.DataFrame(rows)
    print(
        f"View consistency computed: {df['task_id'].nunique()} tasks "
        f"({len(df)} test examples) "
        f"(model={model}, attempt={attempt})"
    )
    return df


def aggregate_to_task(view_df):
    """
    Collapse multi-test-example tasks to one row per task_id by averaging
    the view-level metrics across test examples.
    """
    agg = (
        view_df.groupby("task_id")
        .agg(
            n_views=("n_views", "mean"),
            n_unique_grids=("n_unique_grids", "mean"),
            view_level_accuracy=("view_level_accuracy", "mean"),
            top1_share=("top1_share", "mean"),
            top2_share=("top2_share", "mean"),
            top1_top2_gap=("top1_top2_gap", "mean"),
            vote_entropy=("vote_entropy", "mean"),
            top1_correct_all=("top1_correct", "all"),
        )
        .reset_index()
    )
    return agg


if __name__ == "__main__":
    df = compute_view_consistency()
    out = ROOT / "data" / "view_consistency.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

    task_df = aggregate_to_task(df)
    task_out = ROOT / "data" / "view_consistency_task.csv"
    task_df.to_csv(task_out, index=False)
    print(f"Saved {len(task_df)} task-level rows to {task_out}")
