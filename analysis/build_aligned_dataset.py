"""
Build data/aligned_dataset.csv — one row per task (400 rows).

Columns
-------
task_id
varc_correct          bool   - majority-voted ViT prediction matches ground truth
varc_error_type       str    - correct / wrong_size / close_miss / wrong_content
varc_cell_accuracy    float  - cell-level accuracy (NaN if wrong_size)
human_n               int    - number of participants who attempted this task
human_accuracy        float  - fraction who got it correct (last attempt)
human_error_mode      str    - most common error type among human attempts
human_mean_cell_acc   float  - mean cell accuracy on wrong attempts (NaN if all correct)
both_correct          bool
human_only            bool   - human majority correct, VARC wrong
varc_only             bool   - VARC correct, human majority wrong
both_wrong            bool
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.load_data import (
    load_arc_ground_truth,
    load_varc_predictions,
    load_harc_summary,
)
from analysis.error_analysis import (
    classify_error,
    cell_accuracy,
    compute_varc_errors,
    compute_human_errors,
)


def build():
    print("Loading data...")
    ground_truth     = load_arc_ground_truth()
    varc_predictions = load_varc_predictions()
    harc_df          = load_harc_summary()

    varc_errors  = compute_varc_errors(ground_truth, varc_predictions)
    human_errors = compute_human_errors(harc_df, ground_truth)

    # ── VARC: aggregate to task level (all test examples must be correct) ──
    varc_task = (
        varc_errors.groupby("task_id")
        .agg(
            varc_correct=("error_type", lambda x: (x == "correct").all()),
            varc_error_type=("error_type", lambda x: (
                "correct" if (x == "correct").all()
                else x[x != "correct"].mode().iloc[0] if len(x[x != "correct"]) > 0
                else "correct"
            )),
            varc_cell_accuracy=("cell_accuracy", "mean"),
        )
        .reset_index()
    )

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
