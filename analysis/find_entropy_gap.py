"""
找"人高度收敛但VARC极度发散"的题目。
entropy_gap = varc_entropy - human_entropy
输出 gap 最大的前10道题。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from collections import Counter
from analysis.load_data import load_harc_summary, parse_grid

ROOT = Path(__file__).parent.parent


def grid_key(grid):
    if grid is None:
        return None
    return tuple(tuple(row) for row in grid)


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ── Human entropy: distribution of last-attempt answers per task ──────────────
print("Loading H-ARC data...")
harc_df = load_harc_summary()

# 每位参与者只取最后一次尝试
last_attempt = (
    harc_df.sort_values("attempt_number")
    .groupby(["task_id", "hashed_id"])
    .last()
    .reset_index()
)

human_entropy_rows = []
for task_id, group in last_attempt.groupby("task_id"):
    keys = [grid_key(parse_grid(r)) for r in group["test_output_grid"]]
    keys = [k for k in keys if k is not None]
    if not keys:
        continue
    human_entropy_rows.append({
        "task_id": task_id,
        "human_entropy": entropy(Counter(keys)),
        "human_n": len(keys),
    })

human_ent_df = pd.DataFrame(human_entropy_rows)
print(f"Human entropy computed for {len(human_ent_df)} tasks")


# ── VARC entropy: distribution of raw per-attempt predictions ─────────────────
print("Loading VARC raw predictions...")
model = "ARC-1_ViT"
model_dir = ROOT / "VARC_predictions" / "VARC_predictions" / model

varc_entropy_rows = []
task_grids: dict[str, list] = {}

for attempt_dir in sorted(model_dir.iterdir()):
    if not attempt_dir.is_dir():
        continue
    for path in attempt_dir.glob("*_predictions.json"):
        task_id = path.stem.replace("_predictions", "")
        with open(path) as f:
            raw = json.load(f)
        # 只取第一个 test case 的预测（index "0"）
        grids = raw.get("0", [])
        task_grids.setdefault(task_id, []).extend(grids)

for task_id, grids in task_grids.items():
    keys = [grid_key(g) for g in grids if g is not None]
    if not keys:
        continue
    varc_entropy_rows.append({
        "task_id": task_id,
        "varc_entropy": entropy(Counter(keys)),
        "varc_n": len(keys),
    })

varc_ent_df = pd.DataFrame(varc_entropy_rows)
print(f"VARC entropy computed for {len(varc_ent_df)} tasks")


# ── Merge and compute gap ─────────────────────────────────────────────────────
aligned = pd.read_csv(ROOT / "data" / "aligned_dataset.csv")

df = (
    aligned[["task_id", "human_accuracy", "varc_correct", "human_only"]]
    .merge(human_ent_df, on="task_id", how="inner")
    .merge(varc_ent_df,  on="task_id", how="inner")
)

df["entropy_gap"] = df["varc_entropy"] - df["human_entropy"]

top10 = df.nlargest(10, "entropy_gap")

print("\n=== Top 10 tasks: VARC entropy HIGH, Human entropy LOW ===")
print(top10[["task_id", "human_entropy", "varc_entropy", "entropy_gap",
             "human_accuracy", "varc_correct", "human_n", "varc_n"]].to_string(index=False))

print("\n--- task_id list ---")
for tid in top10["task_id"].tolist():
    print(tid)
