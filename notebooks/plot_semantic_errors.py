import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rows = list(csv.DictReader(open(
    '/Users/jasmineeee/Desktop/arc-error-patterns/results/annotations.csv')))

LABELS = ['near_miss', 'wrong_position', 'wrong_structure',
          'partial_rule', 'wrong_rule', 'no_pattern']
COLORS = {'VARC': '#E07B54', 'Human': '#5B8DB8'}

varc_counts  = {l: sum(1 for r in rows if r['varc_error_label']  == l) for l in LABELS}
human_counts = {l: sum(1 for r in rows if r['human_error_label'] == l) for l in LABELS}

n = len(LABELS)
x = np.arange(n)
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))
bars_h = ax.bar(x - w/2, [human_counts[l] for l in LABELS], w,
                label='Human', color=COLORS['Human'], alpha=0.85)
bars_v = ax.bar(x + w/2, [varc_counts[l]  for l in LABELS], w,
                label='VARC',  color=COLORS['VARC'],  alpha=0.85)

for bar in list(bars_h) + list(bars_v):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.07, str(int(h)),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel('Task count (out of 20)', fontsize=11)
ax.set_title('Semantic Error Label Distribution: Human vs. VARC\n(both_wrong sample, n=20)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 14)
ax.legend(fontsize=11)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/jasmineeee/Desktop/arc-error-patterns/results/semantic_error_dist.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("saved semantic_error_dist.png")
