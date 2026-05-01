"""
Render show_task images for 20 selected both-wrong tasks and save to results/.
Reads raw data directly from arc-project.
Run from: arc-error-patterns/notebooks/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RAW = Path("/Users/jasmineeee/Desktop/ARC-project/arc-error-patterns/data/raw")
ARC_DIR  = RAW / "ARC" / "evaluation"
VARC_DIR = RAW / "V-ARC" / "VARC_predictions" / "ARC-1_ViT"
HARC_CSV = RAW / "H-ARC" / "summary_data.csv"
OUT_DIR  = Path("../results")

# ── colour map ────────────────────────────────────────────────────────────────
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
cmap = mcolors.ListedColormap(ARC_COLORS)

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_grid(value):
    if value is None or (isinstance(value, float)):
        return None
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value.startswith("|"):
        rows = [r for r in value.split("|") if r]
        return [[int(c) for c in row] for row in rows]
    return json.loads(value)

def majority_vote(grids):
    counts = {}
    for g in grids:
        key = tuple(tuple(row) for row in g)
        counts[key] = counts.get(key, 0) + 1
    return [list(row) for row in max(counts, key=counts.get)]

def load_varc_predictions():
    all_grids = {}
    for attempt_dir in sorted(VARC_DIR.iterdir()):
        if not attempt_dir.is_dir():
            continue
        for path in attempt_dir.glob("*_predictions.json"):
            task_id = path.stem.replace("_predictions", "")
            with open(path) as f:
                raw = json.load(f)
            if task_id not in all_grids:
                all_grids[task_id] = {}
            for k, grids in raw.items():
                all_grids[task_id].setdefault(int(k), []).extend(grids)
    return {tid: [majority_vote(by_idx[i]) for i in range(len(by_idx))]
            for tid, by_idx in all_grids.items()}

def load_harc():
    df = pd.read_csv(HARC_CSV)
    df = df[df["task_type"] == "evaluation"].copy()
    df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)
    return df

def _draw(ax, grid, title, correct=None):
    g = np.array(grid)
    ax.imshow(g, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    color = {True: "green", False: "red"}.get(correct, "black")
    ax.set_title(title, fontsize=8, color=color)
    ax.axis("off")
    h, w = g.shape
    for x in range(w + 1): ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(h + 1): ax.axhline(y - 0.5, color="white", lw=0.5)

def grids_equal(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and (a == b).all()

def render_task(task_id, varc_preds, harc_df, max_humans=6):
    with open(ARC_DIR / f"{task_id}.json") as f:
        task = json.load(f)
    train, test = task["train"], task["test"]
    truth = test[0]["output"]
    varc_pred = varc_preds.get(task_id, [[]])[0] if task_id in varc_preds else None

    sub = harc_df[harc_df["task_id"] == task_id]
    last = sub.sort_values("attempt_number").groupby("hashed_id").last().reset_index()
    wrong = last[last["solved"] == False].head(max_humans)
    human_grids = [g for g in (parse_grid(r["test_output_grid"]) for _, r in wrong.iterrows()) if g]

    n = len(train)
    extras = 2 + (varc_pred is not None)
    n_cols = max(n + extras, len(human_grids)) if human_grids else n + extras
    n_rows = 4 if human_grids else 2

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle(f"Task: {task_id}", fontsize=12, fontweight="bold")

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
        ok = grids_equal(varc_pred, truth)
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
    out_path = OUT_DIR / f"task_full_{task_id}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


if __name__ == "__main__":
    print("Loading VARC predictions...")
    varc_preds = load_varc_predictions()
    print(f"  {len(varc_preds)} tasks loaded")

    print("Loading H-ARC data...")
    harc_df = load_harc()
    print(f"  {len(harc_df)} rows loaded")

    already_selected = [
        "08573cc6", "0934a4d8", "103eff5b", "12eac192", "136b0064",
    ]
    new_15 = [
        # very hard  (human_acc 0.00–0.14)
        "79fb03f4", "e6de6e8f", "5b692c0f",
        # hard       (0.167–0.286)
        "891232d6", "ad7e01d0", "94133066", "37d3e8b2",
        # moderate   (0.30–0.375)
        "dc2aa30b", "c92b942c", "73c3b0d8",
        # medium-hard (0.40–0.462)
        "2a5f8217", "85fa5666", "e5c44e8f",
        # medium     (0.50)
        "17cae0c1", "dd2401ed",
    ]
    all_20 = already_selected + new_15

    print(f"\nRendering {len(all_20)} tasks...")
    for tid in all_20:
        render_task(tid, varc_preds, harc_df)

    print("\nDone.")
