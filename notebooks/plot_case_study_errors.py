import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CASE_TASKS = ['ad7e01d0', 'e6de6e8f', '891232d6']

# Error labels for each case study task
VARC_LABELS  = ['wrong_position', 'near_miss', 'wrong_rule']
HUMAN_LABELS = ['partial_rule',   'near_miss', 'partial_rule']

# Ordered error severity: near_miss → wrong_position → partial_rule → wrong_rule
LABEL_ORDER = ['near_miss', 'wrong_position', 'wrong_structure',
               'partial_rule', 'wrong_rule']

ERROR_COLORS = {
    'near_miss':      '#6BAE75',
    'wrong_position': '#F5C542',
    'wrong_structure':'#F0914A',
    'partial_rule':   '#5B8DB8',
    'wrong_rule':     '#C0392B',
}

AGENT_COLORS = {'VARC': '#E07B54', 'Human': '#5B8DB8'}

# ── Build stacked data ────────────────────────────────────────────────────────

def stack_for(label_list):
    """Return ordered list of (error_label, [task_ids]) for stacking."""
    from collections import defaultdict
    groups = defaultdict(list)
    for task, label in zip(CASE_TASKS, label_list):
        groups[label].append(task)
    return [(lbl, groups[lbl]) for lbl in LABEL_ORDER if lbl in groups]

varc_stack  = stack_for(VARC_LABELS)
human_stack = stack_for(HUMAN_LABELS)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5.5))

bar_width = 0.38
positions = {'VARC': 0, 'Human': 1}

def draw_stacked(ax, pos, stack, bar_width):
    bottom = 0
    for label, tasks in stack:
        h = len(tasks)
        bar = ax.bar(pos, h, bar_width, bottom=bottom,
                     color=ERROR_COLORS[label], alpha=0.88,
                     edgecolor='white', linewidth=0.8)
        # annotate task id inside segment
        for i, task in enumerate(tasks):
            y_center = bottom + i + 0.5
            ax.text(pos, y_center, task[:8], ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        bottom += h

draw_stacked(ax, positions['VARC'],  varc_stack,  bar_width)
draw_stacked(ax, positions['Human'], human_stack, bar_width)

# ── Axes labels ───────────────────────────────────────────────────────────────

ax.set_xticks([0, 1])
ax.set_xticklabels(['VARC', 'Human'], fontsize=13, fontweight='bold')
ax.set_ylabel('Number of case-study tasks (n=3)', fontsize=11)
ax.set_ylim(0, 4)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_title('Error Type Distribution: 3 Case Study Tasks\nVARC vs. Human',
             fontsize=13, fontweight='bold', pad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# ── Legend ────────────────────────────────────────────────────────────────────

legend_labels = sorted(
    set(VARC_LABELS + HUMAN_LABELS),
    key=lambda l: LABEL_ORDER.index(l)
)
legend_patches = [
    mpatches.Patch(color=ERROR_COLORS[l], label=l.replace('_', ' '))
    for l in legend_labels
]
ax.legend(handles=legend_patches, title='Error Type', fontsize=10,
          title_fontsize=10, loc='upper right', framealpha=0.85)

plt.tight_layout()
out_path = '/Users/jasmineeee/Desktop/arc-error-patterns/results/case_study_error_dist.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved {out_path}")
