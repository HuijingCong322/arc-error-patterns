"""
Core error metrics and comparison logic.

Error taxonomy
--------------
correct       – exact match with ground truth
wrong_size    – output dimensions don't match (rows or cols differ)
close_miss    – right size, cell accuracy >= 0.8
wrong_content – right size, cell accuracy < 0.8
"""

import pandas as pd


# ── Grid-level primitives ─────────────────────────────────────────────────────

def exact_match(pred, truth):
    if len(pred) != len(truth):
        return False
    return all(list(r1) == list(r2) for r1, r2 in zip(pred, truth))


def size_match(pred, truth):
    if len(pred) != len(truth):
        return False
    return all(len(r1) == len(r2) for r1, r2 in zip(pred, truth))


def cell_accuracy(pred, truth):
    """Fraction of cells matching. Returns None when sizes differ."""
    if not size_match(pred, truth):
        return None
    total = sum(len(r) for r in truth)
    if total == 0:
        return 0.0
    correct = sum(c1 == c2 for r1, r2 in zip(pred, truth) for c1, c2 in zip(r1, r2))
    return correct / total


def classify_error(pred, truth):
    if exact_match(pred, truth):
        return "correct"
    if not size_match(pred, truth):
        return "wrong_size"
    return "close_miss" if cell_accuracy(pred, truth) >= 0.8 else "wrong_content"


# ── VARC error table ──────────────────────────────────────────────────────────

def compute_varc_errors(ground_truth, varc_predictions):
    """
    Returns DataFrame with one row per (task_id, test_idx).
    Columns: task_id, test_idx, error_type, cell_accuracy
    """
    rows = []
    for task_id, truths in ground_truth.items():
        preds = varc_predictions.get(task_id)
        if preds is None:
            continue
        for i, (truth, pred) in enumerate(zip(truths, preds)):
            rows.append({
                "task_id":       task_id,
                "test_idx":      i,
                "error_type":    classify_error(pred, truth),
                "cell_accuracy": cell_accuracy(pred, truth),
            })
    return pd.DataFrame(rows)


# ── Human error table ─────────────────────────────────────────────────────────

def compute_human_errors(
    harc_df,
    ground_truth,
    task_col="task_id",
    participant_col="participant_id",
    attempt_col="attempt",
    grid_col="response_grid",
):
    """
    Uses each participant's LAST attempt (most comparable to VARC single prediction).
    Returns DataFrame with one row per (task_id, participant_id).
    Columns: task_id, participant_id, error_type, cell_accuracy
    """
    from .load_data import parse_grid

    last = (
        harc_df.sort_values(attempt_col)
        .groupby([task_col, participant_col], sort=False)
        .last()
        .reset_index()
    )

    rows = []
    for _, row in last.iterrows():
        task_id = row[task_col]
        truths = ground_truth.get(task_id)
        if truths is None:
            continue
        pred = parse_grid(row[grid_col])
        if pred is None:
            continue
        truth = truths[0]   # evaluation tasks have one test example
        rows.append({
            "task_id":        task_id,
            "participant_id": row[participant_col],
            "error_type":     classify_error(pred, truth),
            "cell_accuracy":  cell_accuracy(pred, truth),
        })
    return pd.DataFrame(rows)


# ── Task-level comparison ─────────────────────────────────────────────────────

def task_level_summary(human_errors, varc_errors):
    """
    Per-task accuracy for both sources.
    Returns DataFrame: task_id, human_accuracy, varc_correct, agreement
    """
    human_acc = (
        human_errors.groupby("task_id")["error_type"]
        .apply(lambda x: (x == "correct").mean())
        .rename("human_accuracy")
    )
    varc_correct = (
        varc_errors.set_index("task_id")["error_type"]
        .eq("correct")
        .rename("varc_correct")
    )
    summary = pd.concat([human_acc, varc_correct], axis=1).dropna().reset_index()
    # agreement: both right or both wrong
    summary["agreement"] = (
        (summary["human_accuracy"] > 0.5) == summary["varc_correct"]
    )
    return summary


def error_type_distribution(errors_df):
    """Returns normalized value_counts of error_type column."""
    return errors_df["error_type"].value_counts(normalize=True).rename("proportion")
