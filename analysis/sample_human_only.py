"""
随机抽取"人对VARC错"的任务可视化。
每次抽10道，共抽3次，输出到 results/sample_run_1/, _2/, _3/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import pandas as pd

from analysis.load_data import (
    load_arc_ground_truth, load_varc_predictions,
    load_harc_summary, parse_grid,
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
ground_truth     = load_arc_ground_truth()
varc_predictions = load_varc_predictions()
harc_df          = load_harc_summary()
aligned          = pd.read_csv("data/aligned_dataset.csv")

# "人对VARC错" = human_only group
human_only_ids = aligned[aligned["human_only"] == True]["task_id"].tolist()
print(f"human_only tasks: {len(human_only_ids)}")

# ── Visualization helpers ─────────────────────────────────────────────────────
ARC_COLORS = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00",
              "#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25"]
cmap = mcolors.ListedColormap(ARC_COLORS)

def _draw(ax, grid, title, correct=None):
    g = np.array(grid)
    ax.imshow(g, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    color = {True: "green", False: "red"}.get(correct, "black")
    ax.set_title(title, fontsize=8, color=color)
    ax.axis("off")
    h, w = g.shape
    for x in range(w + 1): ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(h + 1): ax.axhline(y - 0.5, color="white", lw=0.5)

def _grids_equal(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and (a == b).all()

def save_task(task_id, save_path, max_humans=6):
    with open(f"ARC-AGI/data/evaluation/{task_id}.json") as f:
        task = json.load(f)
    train, test = task["train"], task["test"]
    truth = test[0]["output"]
    varc_pred = varc_predictions.get(task_id, [None])[0]

    # wrong human submissions
    sub = harc_df[harc_df["task_id"] == task_id]
    last = sub.sort_values("attempt_number").groupby("hashed_id").last().reset_index()
    wrong = last[last["solved"] == False].head(max_humans)
    human_grids = [g for g in (parse_grid(r["test_output_grid"]) for _, r in wrong.iterrows()) if g]

    # human accuracy for subtitle
    human_acc = aligned.loc[aligned["task_id"] == task_id, "human_accuracy"].values[0]

    n = len(train)
    extras = 2 + (varc_pred is not None)
    n_cols = max(n + extras, len(human_grids)) if human_grids else n + extras
    n_rows = 4 if human_grids else 2

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle(
        f"Task: {task_id}   |   Human accuracy: {human_acc:.0%}   |   VARC: wrong",
        fontsize=11, fontweight="bold"
    )

    r1 = [fig.add_subplot(n_rows, n_cols, i + 1)          for i in range(n_cols)]
    r2 = [fig.add_subplot(n_rows, n_cols, n_cols + i + 1) for i in range(n_cols)]
    for ax in r1: ax.axis("off")
    for ax in r2: ax.axis("off")

    for i, ex in enumerate(train):
        _draw(r1[i], ex["input"],  f"Train {i+1} In")
        _draw(r2[i], ex["output"], f"Train {i+1} Out")

    col = n
    _draw(r1[col], test[0]["input"], "Test Input")
    r2[col].set_title("Test Output", fontsize=8); col += 1
    _draw(r1[col], truth, "Ground Truth"); col += 1

    if varc_pred is not None:
        ok = _grids_equal(varc_pred, truth)
        _draw(r1[col], varc_pred, f"VARC ({'OK' if ok else 'X'})", correct=ok)

    if human_grids:
        fig.text(0.01, 0.49, "Wrong human submissions (last attempt):", fontsize=8)
        r3 = [fig.add_subplot(n_rows, n_cols, 2*n_cols + i + 1) for i in range(n_cols)]
        r4 = [fig.add_subplot(n_rows, n_cols, 3*n_cols + i + 1) for i in range(n_cols)]
        for ax in r3: ax.axis("off")
        for ax in r4: ax.axis("off")
        for i, g in enumerate(human_grids):
            _draw(r3[i], g, f"Human {i+1}", correct=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

# ── Sample and save ───────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
N_SAMPLES = 10
N_RUNS = 3

# 一次性从122道题中不放回抽取30道，再按顺序分给3个run，保证完全不重复
all_sampled = pd.Series(human_only_ids).sample(N_SAMPLES * N_RUNS, random_state=42).tolist()

for run in range(1, N_RUNS + 1):
    out_dir = RESULTS_DIR / f"sample_run_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = all_sampled[(run - 1) * N_SAMPLES : run * N_SAMPLES]
    print(f"\nRun {run}: {sampled}")

    for i, tid in enumerate(sampled, 1):
        save_path = out_dir / f"{i:02d}_{tid}.png"
        save_task(tid, save_path)
        print(f"  [{i}/{N_SAMPLES}] {tid} -> {save_path.name}")

print("\nDone. Output folders:")
for run in range(1, N_RUNS + 1):
    print(f"  results/sample_run_{run}/")
