from pathlib import Path
import csv

# 统计总结H-ARC和V-ARC做对多少做错多少

input_file = Path("data/processed/aligned_dataset.csv")
output_file = Path("results/aligned_summary.txt")

total = 0
arc_equals_harc_true = 0
arc_equals_varc_true = 0
both_true = 0
both_false = 0
harc_only_true = 0
varc_only_true = 0

with input_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1

        arc_equals_harc = row["arc_equals_harc"] == "True"
        arc_equals_varc = row["arc_equals_varc"] == "True"

        if arc_equals_harc:
            arc_equals_harc_true += 1
        if arc_equals_varc:
            arc_equals_varc_true += 1

        if arc_equals_harc and arc_equals_varc:
            both_true += 1
        elif (not arc_equals_harc) and (not arc_equals_varc):
            both_false += 1
        elif arc_equals_harc and (not arc_equals_varc):
            harc_only_true += 1
        elif (not arc_equals_harc) and arc_equals_varc:
            varc_only_true += 1

lines = [
    f"Total tasks: {total}",
    f"H-ARC matches ARC official: {arc_equals_harc_true}",
    f"VARC matches ARC official: {arc_equals_varc_true}",
    "",
    f"Both H-ARC and VARC correct: {both_true}",
    f"Both H-ARC and VARC wrong: {both_false}",
    f"Only H-ARC correct: {harc_only_true}",
    f"Only VARC correct: {varc_only_true}",
]

for line in lines:
    print(line)

with output_file.open("w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print()
print(f"Saved summary to {output_file}")