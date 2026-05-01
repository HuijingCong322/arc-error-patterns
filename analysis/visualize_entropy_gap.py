"""
可视化"人类高度收敛、VARC极度发散"的分析结果。
生成：
  results/entropy_gap_scatter.png      — 所有400道题的散点图
  results/entropy_gap_top3_*.png       — 前3道题的详细可视化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

from analysis.load_data import load_harc_summary, load_arc_ground_truth, parse_grid

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

ARC_COLORS = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00",
              "#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25"]
cmap = mcolors.ListedColormap(ARC_COLORS)

random.seed(42)


# ── helpers ───────────────────────────────────────────────────────────────────

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

def draw_grid(ax, grid, title, title_color="black"):
    g = np.array(grid)
    ax.imshow(g, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=7, color=title_color, pad=2)
    ax.axis("off")
    h, w = g.shape
    for x in range(w + 1): ax.axvline(x - 0.5, color="white", lw=0.4)
    for y in range(h + 1): ax.axhline(y - 0.5, color="white", lw=0.4)

def grids_equal(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and (a == b).all()


# ── load raw VARC predictions (no majority vote) ──────────────────────────────

def load_varc_raw(model="ARC-1_ViT"):
    model_dir = ROOT / "VARC_predictions" / "VARC_predictions" / model
    task_grids: dict[str, list] = {}
    for attempt_dir in sorted(model_dir.iterdir()):
        if not attempt_dir.is_dir():
            continue
        for path in attempt_dir.glob("*_predictions.json"):
            task_id = path.stem.replace("_predictions", "")
            with open(path) as f:
                raw = json.load(f)
            grids = raw.get("0", [])
            task_grids.setdefault(task_id, []).extend(grids)
    return task_grids


# ── recompute entropy tables ──────────────────────────────────────────────────

print("Loading data...")
harc_df = load_harc_summary()
ground_truth = load_arc_ground_truth()
varc_raw = load_varc_raw()
aligned = pd.read_csv(ROOT / "data" / "aligned_dataset.csv")

last_attempt = (
    harc_df.sort_values("attempt_number")
    .groupby(["task_id", "hashed_id"]).last().reset_index()
)

rows = []
for task_id, group in last_attempt.groupby("task_id"):
    hkeys = [grid_key(parse_grid(r)) for r in group["test_output_grid"]]
    hkeys = [k for k in hkeys if k is not None]
    vgrids = varc_raw.get(task_id, [])
    vkeys = [grid_key(g) for g in vgrids if g is not None]
    if not hkeys or not vkeys:
        continue
    rows.append({
        "task_id": task_id,
        "human_entropy": entropy(Counter(hkeys)),
        "varc_entropy":  entropy(Counter(vkeys)),
    })

df = (
    aligned[["task_id","human_accuracy","varc_correct","human_only","both_correct","both_wrong","varc_only"]]
    .merge(pd.DataFrame(rows), on="task_id")
)
df["entropy_gap"] = df["varc_entropy"] - df["human_entropy"]
top10 = df.nlargest(10, "entropy_gap")


# ════════════════════════════════════════════════════════════════════════════
# 图1: 散点图 — human_entropy vs varc_entropy（全400道题）
# ════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 7))

# 四象限着色
palette = {
    "both_correct": ("#2ECC40", "Both Correct"),
    "human_only":   ("#0074D9", "Human Only"),
    "varc_only":    ("#FF851B", "VARC Only"),
    "both_wrong":   ("#FF4136", "Both Wrong"),
}
for col, (color, label) in palette.items():
    sub = df[df[col] == True]
    ax.scatter(sub["human_entropy"], sub["varc_entropy"],
               c=color, label=label, alpha=0.65, s=40, linewidths=0)

# 对角线 y=x
lim = max(df["varc_entropy"].max(), df["human_entropy"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="y = x")

# 标出 top10
for _, row in top10.iterrows():
    ax.annotate(
        row["task_id"][:8],
        (row["human_entropy"], row["varc_entropy"]),
        fontsize=5.5, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points",
        color="#555555",
    )
    ax.scatter(row["human_entropy"], row["varc_entropy"],
               s=80, c="none", edgecolors="black", linewidths=1.2, zorder=5)

ax.set_xlabel("Human Entropy (bits)", fontsize=11)
ax.set_ylabel("VARC Entropy (bits)", fontsize=11)
ax.set_title("Answer Entropy: Humans vs. VARC\n(circled = top-10 entropy gap)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(-0.3, lim)
ax.set_ylim(-0.3, lim)

# 象限标签
ax.text(0.02, 0.97, "Humans converge\nVARC diverges", transform=ax.transAxes,
        fontsize=8, color="#0074D9", va="top", style="italic")
ax.text(0.75, 0.03, "Both converge", transform=ax.transAxes,
        fontsize=8, color="#2ECC40", va="bottom", style="italic")

plt.tight_layout()
scatter_path = RESULTS / "entropy_gap_scatter.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {scatter_path}")


# ════════════════════════════════════════════════════════════════════════════
# 图2–4: 前3道题（human entropy ≈ 0）的详细对比图
# ════════════════════════════════════════════════════════════════════════════

top3 = top10[top10["human_entropy"] < 0.01].head(3)

for _, task_row in top3.iterrows():
    task_id = task_row["task_id"]
    print(f"Rendering {task_id}...")

    # 读取任务 JSON
    task_path = ROOT / "ARC-AGI" / "data" / "evaluation" / f"{task_id}.json"
    with open(task_path) as f:
        task = json.load(f)
    train_examples = task["train"]
    test_input     = task["test"][0]["input"]
    truth          = ground_truth[task_id][0]

    # 收集不重复的 VARC 预测（最多展示 N_VARC 个）
    N_VARC = 8
    all_varc = [g for g in varc_raw.get(task_id, []) if g is not None]
    unique_varc = list({grid_key(g): g for g in all_varc}.values())
    random.shuffle(unique_varc)
    sample_varc = unique_varc[:N_VARC]
    n_varc_unique = len(unique_varc)

    # 收集人类最后提交（应全部正确）
    sub = last_attempt[last_attempt["task_id"] == task_id]
    human_grids = [parse_grid(r) for r in sub["test_output_grid"]]
    human_grids = [g for g in human_grids if g is not None]
    human_acc   = task_row["human_accuracy"]
    varc_ent    = task_row["varc_entropy"]

    n_train = len(train_examples)
    # layout: rows = [train_in, train_out, test+truth+varc, human_row]
    # cols = max(n_train, 2 + N_VARC)
    n_cols = max(n_train + 2, N_VARC)  # at least enough for varc row
    n_rows = 4

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.4))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle(
        f"Task {task_id}  |  Human accuracy: {human_acc:.0%}  |  "
        f"Human entropy: ~0  |  VARC entropy: {varc_ent:.1f} bits  |  "
        f"Unique VARC predictions: {n_varc_unique} / {len(all_varc)}",
        fontsize=10, fontweight="bold", y=0.99,
    )

    def cell(row_i, col_i):
        return fig.add_subplot(n_rows, n_cols, row_i * n_cols + col_i + 1)

    # ── Row 0+1: training examples ──────────────────────────────────────────
    for i, ex in enumerate(train_examples):
        ax = cell(0, i)
        draw_grid(ax, ex["input"], f"Train {i+1} In")
        ax = cell(1, i)
        draw_grid(ax, ex["output"], f"Train {i+1} Out")

    # blank remaining train slots
    for i in range(n_train, n_cols):
        cell(0, i).axis("off")
        cell(1, i).axis("off")

    # ── Row 2: test input | ground truth | VARC samples ────────────────────
    ax = cell(2, 0)
    draw_grid(ax, test_input, "Test Input")

    ax = cell(2, 1)
    draw_grid(ax, truth, "Ground Truth", title_color="green")

    for j, vg in enumerate(sample_varc):
        ax = cell(2, 2 + j)
        ok = grids_equal(vg, truth)
        draw_grid(ax, vg, f"VARC sample {j+1}", title_color="green" if ok else "red")

    for j in range(2 + len(sample_varc), n_cols):
        cell(2, j).axis("off")

    # row 2 section label
    fig.text(0.01, 0.52,
             f"▼ {n_varc_unique} unique VARC predictions (showing {len(sample_varc)})",
             fontsize=8, color="#CC0000", style="italic")

    # ── Row 3: human answers ────────────────────────────────────────────────
    unique_human = list({grid_key(g): g for g in human_grids}.values())
    n_show_human = min(len(unique_human), n_cols)

    for j in range(n_cols):
        ax = cell(3, j)
        if j < n_show_human:
            g = unique_human[j]
            ok = grids_equal(g, truth)
            draw_grid(ax, g, f"Human {j+1}", title_color="green" if ok else "red")
        else:
            ax.axis("off")

    fig.text(0.01, 0.26,
             f"▼ Human answers ({len(human_grids)} participants, "
             f"{len(unique_human)} unique — showing {n_show_human})",
             fontsize=8, color="#0055CC", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = RESULTS / f"entropy_gap_top3_{task_id}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

print("\nAll done.")
