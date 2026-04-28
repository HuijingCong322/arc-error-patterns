"""
Build data/aligned_dataset.csv — one row per task (400 rows).

Columns
-------
task_id
varc_correct          bool   - official plain VARC pass@2 correctness on attempt_0
varc_error_type       str    - top-1 prediction error type (correct / wrong_size / ...)
varc_cell_accuracy    float  - top-1 prediction cell-level accuracy (NaN if wrong_size)
human_n               int    - number of participants who attempted this task
human_accuracy        float  - fraction who got it correct (last attempt)
human_error_mode      str    - most common error type among human attempts
human_mean_cell_acc   float  - mean cell accuracy on wrong attempts (NaN if all correct)
both_correct          bool
human_only            bool   - human majority correct, VARC wrong
varc_only             bool   - VARC correct, human majority wrong
both_wrong            bool
n_views               float  - mean view count across test examples (≈510)
n_unique_grids        float  - mean number of distinct prediction grids across views
view_level_accuracy   float  - mean fraction of views matching ground truth
top1_share            float  - mean top-1 vote share (top-1 count / n_views)
top2_share            float  - mean top-2 vote share
top1_top2_gap         float  - mean (top1 - top2) / n_views — model's top-1 confidence margin
vote_entropy          float  - mean Shannon entropy (nats) over view votes
human_n_responses     int    - participant count contributing answers
human_n_unique_grids  int    - number of distinct human final-answer grids
human_top1_share      float  - most-common human answer's share of responses
human_top1_correct    bool   - whether the human top-1 answer is the correct one
human_top1_top2_gap   float  - gap between top-1 and top-2 human answer counts
human_vote_entropy    float  - Shannon entropy (nats) over human answer distribution
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.load_data import (
    load_arc_ground_truth,
    load_varc_predictions,
    load_varc_vote_entries,
    load_harc_summary,
)
from analysis.error_analysis import (
    classify_error,
    compute_varc_errors,
    compute_human_errors,
)
from analysis.view_consistency import (
    compute_view_consistency,
    aggregate_to_task as aggregate_view_to_task,
)
from analysis.human_consistency import compute_human_consistency


def build():
    print("Loading data...")
    ground_truth     = load_arc_ground_truth()
    varc_predictions = load_varc_predictions()
    varc_votes       = load_varc_vote_entries(top_k=2)
    harc_df          = load_harc_summary()

    varc_errors  = compute_varc_errors(ground_truth, varc_predictions)
    human_errors = compute_human_errors(harc_df, ground_truth)

    # ── VARC: top-1 error profile for visualization / error typing ─────────
    varc_top1_task = (
        varc_errors.groupby("task_id")
        .agg(
            varc_error_type=("error_type", lambda x: (
                "correct" if (x == "correct").all()
                else x[x != "correct"].mode().iloc[0] if len(x[x != "correct"]) > 0
                else "correct"
            )),
            varc_cell_accuracy=("cell_accuracy", "mean"),
        )
        .reset_index()
    )

    # ── VARC: official plain VARC correctness (attempt_0, top-2 / pass@2) ──
    varc_pass2_rows = []
    for task_id, truths in ground_truth.items():
        votes_by_idx = varc_votes.get(task_id)
        if votes_by_idx is None:
            continue
        per_test_correct = []
        for test_idx, truth in enumerate(truths):
            candidates = votes_by_idx.get(test_idx, [])
            top2_correct = any(
                classify_error(entry["prediction"], truth) == "correct"
                for entry in candidates[:2]
            )
            per_test_correct.append(top2_correct)
        varc_pass2_rows.append({
            "task_id": task_id,
            "varc_correct": all(per_test_correct),
        })
    varc_pass2_task = pd.DataFrame(varc_pass2_rows)
    varc_task = pd.merge(varc_top1_task, varc_pass2_task, on="task_id", how="inner")

    # ── VARC: view-consistency metrics over the 510 random views ───────────
    view_df = compute_view_consistency(ground_truth=ground_truth)
    view_task = aggregate_view_to_task(view_df).drop(
        columns=["top1_correct_all"]
    )
    varc_task = pd.merge(varc_task, view_task, on="task_id", how="left")

    # ── Human: aggregate to task level ────────────────────────────────────
    def human_error_mode(x):
        vc = x.value_counts()
        return vc.index[0] if len(vc) > 0 else None

    def mean_cell_acc_wrong(group):
        wrong = group[group["error_type"] != "correct"]["cell_accuracy"].dropna()
        return wrong.mean() if len(wrong) > 0 else float("nan")

    human_task = (
        human_errors.groupby("task_id")
        .apply(lambda g: pd.Series({
            "human_n":             len(g),
            "human_accuracy":      (g["error_type"] == "correct").mean(),
            "human_error_mode":    human_error_mode(g["error_type"]),
            "human_mean_cell_acc": mean_cell_acc_wrong(g),
        }))
        .reset_index()
    )

    # ── Human: answer-distribution metrics (mirrors VARC view-consistency) ─
    human_consistency = compute_human_consistency(harc_df=harc_df, ground_truth=ground_truth)
    human_task = pd.merge(human_task, human_consistency, on="task_id", how="left")

    # ── Merge ──────────────────────────────────────────────────────────────
    df = pd.merge(varc_task, human_task, on="task_id", how="inner")

    human_majority = df["human_accuracy"] > 0.5
    df["both_correct"] = human_majority  & df["varc_correct"]
    df["human_only"]   = human_majority  & ~df["varc_correct"]
    df["varc_only"]    = ~human_majority & df["varc_correct"]
    df["both_wrong"]   = ~human_majority & ~df["varc_correct"]

    # ── Save ───────────────────────────────────────────────────────────────
    out = ROOT / "data" / "aligned_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df[["both_correct","human_only","varc_only","both_wrong"]].sum().to_string())
    return df


if __name__ == "__main__":
    build()
