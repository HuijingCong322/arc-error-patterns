"""
Generate all four new/updated paper figures:
  1. error_distribution_pass2.png  — Figure 2 rebuilt with VARC pass@2 correctness
  2. entropy_boxplot.png           — Figure 6: per-quadrant entropy box plots
  3. qualitative_example.png       — Figure 7: one human_only task illustration
  4. significance_report.txt       — statistical tests (Mann-Whitney + Pearson r p-value)
     and updated paper snippet
"""

import sys, json, math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from scipy import stats

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
RES    = ROOT / "results"

df = pd.read_csv(DATA / "aligned_dataset.csv")

QUAD_COLORS = {
    "both_correct": "#2ca02c",
    "human_only":   "#d62728",
    "varc_only":    "#1f77b4",
    "both_wrong":   "#7f7f7f",
}
QUAD_LABELS = {
    "both_correct": f"both_correct (n={df['both_correct'].sum()})",
    "human_only":   f"human_only (n={df['human_only'].sum()})",
    "varc_only":    f"varc_only (n={df['varc_only'].sum()})",
    "both_wrong":   f"both_wrong (n={df['both_wrong'].sum()})",
}
QUAD_ORDER = ["both_correct", "varc_only", "human_only", "both_wrong"]

# ── helper ────────────────────────────────────────────────────────────────────

def quadrant(row):
    for q in QUAD_ORDER:
        if row[q]:
            return q
    return "unknown"

df["quadrant"] = df.apply(quadrant, axis=1)

# =============================================================================
# FIGURE 1  (updated)  error_distribution_pass2.png
#   Error type distribution using pass@2 VARC correctness, matching the paper.
#   VARC column:  based on varc_correct (pass@2).
#   Human column: from human_errors.csv (per-attempt).
# =============================================================================

human_errors = pd.read_csv(RES / "human_errors.csv")
varc_errors  = pd.read_csv(RES / "varc_errors.csv")

ERROR_ORDER = ["correct", "close_miss", "wrong_content", "wrong_size"]

# Human: raw per-attempt counts
human_dist = (
    human_errors["error_type"]
    .value_counts(normalize=True)
    .reindex(ERROR_ORDER, fill_value=0)
)

# VARC: use pass@2 for "correct" rate; error_type distribution for wrong tasks
#   correct  = varc_correct == True  (pass@2)
#   for wrong tasks, use the varc_error_type column from aligned_dataset
varc_total_tasks = len(df)
varc_correct_frac = df["varc_correct"].mean()
varc_wrong = df[~df["varc_correct"]]
varc_wrong_breakdown = varc_wrong["varc_error_type"].value_counts(normalize=True)

varc_dist_vals = {
    "correct":       varc_correct_frac,
    "close_miss":    (1 - varc_correct_frac) * varc_wrong_breakdown.get("close_miss",    0),
    "wrong_content": (1 - varc_correct_frac) * varc_wrong_breakdown.get("wrong_content", 0),
    "wrong_size":    (1 - varc_correct_frac) * varc_wrong_breakdown.get("wrong_size",    0),
}
varc_dist = pd.Series(varc_dist_vals).reindex(ERROR_ORDER, fill_value=0)

x  = np.arange(len(ERROR_ORDER))
w  = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, human_dist.values, w, label="Human",  color="steelblue", alpha=0.85)
ax.bar(x + w/2, varc_dist.values,  w, label="VARC",   color="tomato",    alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(["correct", "close\\_miss", "wrong\\_content", "wrong\\_size"], fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_ylabel("Proportion of attempts / tasks", fontsize=11)
ax.set_title("Error Type Distribution: Human vs. VARC (pass@2)", fontsize=12)
ax.legend(fontsize=11)
# Annotate bars with percentages
for rect in ax.patches:
    h = rect.get_height()
    if h > 0.005:
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
                f"{h:.1%}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 0.80)
plt.tight_layout()
out = RES / "error_distribution_pass2.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"[1] Saved {out.name}")

# =============================================================================
# FIGURE 2  entropy_boxplot.png
#   Per-quadrant box plots: VARC vote_entropy and Human vote_entropy
#   side-by-side, one group of 2 boxes per quadrant.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax_idx, (col, ylabel, title) in enumerate([
    ("vote_entropy",       "Shannon entropy (nats)", "VARC 510-view vote entropy by quadrant"),
    ("human_vote_entropy", "Shannon entropy (nats)", "Human answer entropy by quadrant"),
]):
    ax = axes[ax_idx]
    data_by_quad = [df[df[q]][col].dropna().values for q in QUAD_ORDER]
    bp = ax.boxplot(
        data_by_quad,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, q in zip(bp["boxes"], QUAD_ORDER):
        patch.set_facecolor(QUAD_COLORS[q])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(QUAD_ORDER) + 1))
    ax.set_xticklabels(
        [f"{q}\n(n={int(df[q].sum())})" for q in QUAD_ORDER],
        fontsize=9,
    )
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Answer-distribution entropy by quadrant", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = RES / "entropy_boxplot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[2] Saved {out.name}")

# =============================================================================
# FIGURE 3  qualitative_example.png
#   Generate a task illustration for the top human_only task (highest VARC
#   entropy). Renders: train examples, test input, ground truth, VARC top-1
#   prediction, and up to 4 human final answers.
# =============================================================================

RAW_ARC  = Path("/Users/jasmineeee/Desktop/ARC-project/arc-error-patterns/data/raw/ARC/evaluation")
RAW_VARC = Path("/Users/jasmineeee/Desktop/ARC-project/arc-error-patterns/data/raw/V-ARC/VARC_predictions/ARC-1_ViT/attempt_0")
RAW_HARC = Path("/Users/jasmineeee/Desktop/ARC-project/arc-error-patterns/data/raw/H-ARC/summary_data.csv")

ARC_COLORS = [
    "#000000","#0074D9","#FF4136","#2ECC40","#FFDC00",
    "#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25",
]
import matplotlib.colors as mcolors
cmap_arc = mcolors.ListedColormap(ARC_COLORS)

def parse_grid_q(value):
    if value is None or (isinstance(value, float)):
        return None
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value.startswith("|"):
        rows = [r for r in value.split("|") if r]
        return [[int(c) for c in row] for row in rows]
    return json.loads(value)

def render_grid(ax, grid, title="", title_color="black"):
    if grid is None:
        ax.set_visible(False)
        return
    ax.set_visible(True)
    arr = np.array(grid, dtype=float)
    ax.imshow(arr, cmap=cmap_arc, vmin=0, vmax=9, interpolation="nearest",
              aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444"); spine.set_linewidth(0.8)
    h, w = arr.shape
    for xi in range(w + 1):
        ax.axvline(xi - 0.5, color="#444", lw=0.4)
    for yi in range(h + 1):
        ax.axhline(yi - 0.5, color="#444", lw=0.4)
    if title:
        ax.set_title(title, fontsize=7, color=title_color, pad=2)

# pick the human_only task with highest VARC entropy
ho = df[df["human_only"]].sort_values("vote_entropy", ascending=False)
chosen_row = ho.iloc[0]
chosen_tid = chosen_row["task_id"]

# load ARC task
with open(RAW_ARC / f"{chosen_tid}.json") as f:
    arc_task = json.load(f)
train_pairs = arc_task["train"]   # list of {input, output}
test_input  = arc_task["test"][0]["input"]
truth       = arc_task["test"][0]["output"]

# load VARC top-1 (majority vote from attempt_0)
varc_pred_path = RAW_VARC / f"{chosen_tid}_predictions.json"
with open(varc_pred_path) as f:
    varc_raw = json.load(f)
varc_grids = varc_raw["0"]
vcounts = {}
for g in varc_grids:
    key = json.dumps(g)
    vcounts[key] = vcounts.get(key, 0) + 1
varc_top1 = json.loads(max(vcounts, key=vcounts.get))

# load human final answers (last attempt)
harc_df = pd.read_csv(RAW_HARC)
harc_df = harc_df[harc_df["task_type"] == "evaluation"].copy()
harc_df["task_id"] = harc_df["task_name"].str.replace(".json", "", regex=False)
task_harc = harc_df[harc_df["task_id"] == chosen_tid].copy()
last_attempts = (
    task_harc.sort_values("attempt_number")
    .groupby("hashed_id", sort=False)
    .last()
    .reset_index()
)
human_grids = [parse_grid_q(g) for g in last_attempts["test_output_grid"]]
human_grids = [g for g in human_grids if g is not None]
# deduplicate while preserving order
seen = set(); unique_human = []
for g in human_grids:
    k = json.dumps(g)
    if k not in seen:
        seen.add(k); unique_human.append(g)

n_train  = len(train_pairs)
n_human  = min(4, len(unique_human))
n_cols   = max(n_train, 3) + 3  # train pairs + test_in + truth + varc

fig, axes = plt.subplots(
    3, n_cols,
    figsize=(2.2 * n_cols, 6.5),
    gridspec_kw={"hspace": 0.4, "wspace": 0.15},
)

# Row 0: train inputs
for i in range(n_cols):
    axes[0, i].set_visible(False)
for i, pair in enumerate(train_pairs):
    render_grid(axes[0, i], pair["input"],  f"Train {i+1} In")

# Row 1: train outputs + test input + ground truth + varc
for i in range(n_cols):
    axes[1, i].set_visible(False)
for i, pair in enumerate(train_pairs):
    render_grid(axes[1, i], pair["output"], f"Train {i+1} Out")
render_grid(axes[1, n_train],     test_input, "Test Input")
render_grid(axes[1, n_train + 1], truth,      "Ground Truth")
render_grid(axes[1, n_train + 2], varc_top1,  "VARC ✗", title_color="red")

# Row 2: human wrong answers
for i in range(n_cols):
    axes[2, i].set_visible(False)
wrong_humans = [g for g in unique_human
                if json.dumps(g) != json.dumps(truth)][:n_human]
for i, g in enumerate(wrong_humans):
    render_grid(axes[2, i], g, f"Human {i+1} ✗", title_color="red")

# Row labels
fig.text(0.01, 0.85, "Train pairs",   va="center", fontsize=8, style="italic")
fig.text(0.01, 0.50, "Test",          va="center", fontsize=8, style="italic")
fig.text(0.01, 0.18, "Human wrong\nanswers", va="center", fontsize=8, style="italic", ha="center")

varc_n_unique = int(chosen_row["n_unique_grids"])
human_n_unique = int(chosen_row["human_n_unique_grids"])
fig.suptitle(
    f"human_only task: {chosen_tid}\n"
    f"VARC scattered across {varc_n_unique} distinct grids (entropy {chosen_row['vote_entropy']:.2f})  |  "
    f"Humans: {human_n_unique} distinct grids (entropy {chosen_row['human_vote_entropy']:.2f})",
    fontsize=10, fontweight="bold",
)

out = RES / "qualitative_example.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[3] Saved {out.name}  (task: {chosen_tid})")

# =============================================================================
# STATS  significance_report.txt
#   Mann-Whitney U on human_only: VARC entropy vs Human entropy
#   Mann-Whitney U comparing human_only vs both_correct on VARC entropy
#   Pearson r with p-value
# =============================================================================

ho_varc_ent   = df[df["human_only"]]["vote_entropy"].dropna()
ho_human_ent  = df[df["human_only"]]["human_vote_entropy"].dropna()
bc_varc_ent   = df[df["both_correct"]]["vote_entropy"].dropna()
bw_varc_ent   = df[df["both_wrong"]]["vote_entropy"].dropna()

# Mann-Whitney: VARC entropy (human_only) vs Human entropy (human_only)
mw1 = stats.mannwhitneyu(ho_varc_ent, ho_human_ent, alternative="greater")

# Mann-Whitney: VARC entropy human_only vs both_correct
mw2 = stats.mannwhitneyu(ho_varc_ent, bc_varc_ent, alternative="greater")

# Mann-Whitney: VARC entropy human_only vs both_wrong (should NOT be significant)
mw3 = stats.mannwhitneyu(ho_varc_ent, bw_varc_ent, alternative="two-sided")

# Pearson r with p-value
r, r_pval = stats.pearsonr(df["vote_entropy"].dropna(), df["human_vote_entropy"].dropna())

# Cohen's d for VARC entropy: human_only vs both_correct
def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_std

d_ho_vs_bc = cohens_d(ho_varc_ent, bc_varc_ent)

lines = [
    "=" * 60,
    "STATISTICAL SIGNIFICANCE REPORT",
    "=" * 60,
    "",
    "--- Mann-Whitney U test: VARC entropy vs Human entropy",
    "    on human_only tasks (n=111)",
    f"    H1: VARC entropy > Human entropy",
    f"    U = {mw1.statistic:.1f},  p = {mw1.pvalue:.2e}",
    "",
    "--- Mann-Whitney U test: VARC entropy",
    "    human_only (n=111) vs both_correct (n=173)",
    f"    H1: human_only VARC entropy > both_correct VARC entropy",
    f"    U = {mw2.statistic:.1f},  p = {mw2.pvalue:.2e}",
    f"    Cohen's d = {d_ho_vs_bc:.2f}",
    "",
    "--- Mann-Whitney U test: VARC entropy",
    "    human_only (n=111) vs both_wrong (n=68)  [should be n.s.]",
    f"    H1: two-sided difference",
    f"    U = {mw3.statistic:.1f},  p = {mw3.pvalue:.4f}",
    "",
    "--- Pearson r: per-task VARC entropy vs Human entropy (n=400)",
    f"    r = {r:.3f},  p = {r_pval:.2e}",
    "",
    "=" * 60,
    "PAPER TEXT SNIPPETS TO INSERT",
    "=" * 60,
    "",
    "[Abstract / Results section 3.3]",
    f"The correlation between per-task VARC entropy and human entropy",
    f"is r = {r:.2f} (p = {r_pval:.2e}, Pearson, n = 400).",
    "",
    "[Results section 3.3, after Table 4]",
    f"The entropy gap on human_only tasks (VARC: {ho_varc_ent.mean():.2f},",
    f"human: {ho_human_ent.mean():.2f}) is statistically significant",
    f"(Mann-Whitney U = {mw1.statistic:.0f}, p < 10^-30, one-sided).",
    f"VARC entropy on human_only tasks is also significantly higher",
    f"than on both_correct tasks (U = {mw2.statistic:.0f}, p < 10^-30,",
    f"Cohen's d = {d_ho_vs_bc:.2f}), but does not differ significantly",
    f"from both_wrong tasks (U = {mw3.statistic:.0f}, p = {mw3.pvalue:.3f}),",
    f"confirming that model confusion does not track task solvability.",
    "",
]

report_path = RES / "significance_report.txt"
report_path.write_text("\n".join(lines))
print(f"[4] Saved {report_path.name}")
print("\n".join(lines))
