from pathlib import Path
import json
import csv

# 对同一个track_id分别得到三个ARC的output 最后变成三个二维的list

def parse_harc_grid(grid_str: str):
    rows = [row for row in grid_str.strip().split("|") if row]
    grid = [[int(ch) for ch in row] for row in rows]
    return grid


task_id = "0a1d4ef5"

# ---------- ARC official ----------
arc_file = Path(f"data/raw/ARC/evaluation/{task_id}.json")
with arc_file.open("r", encoding="utf-8") as f:
    arc_data = json.load(f)

arc_output = arc_data["test"][0]["output"]

# ---------- H-ARC ----------
harc_file = Path("data/raw/H-ARC/summary_data.csv")
harc_output = None

with harc_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["task_type"] == "evaluation" and row["task_name"] == f"{task_id}.json":
            harc_output = parse_harc_grid(row["test_output_grid"])
            break

# ---------- VARC ----------
varc_file = Path(f"data/raw/V-ARC/VARC_predictions/ARC-1_Unet/attempt_1/{task_id}_predictions.json")
with varc_file.open("r", encoding="utf-8") as f:
    varc_data = json.load(f)

varc_output = varc_data["0"][0]   # first prediction

print("task_id:", task_id)
print()

print("ARC official output:")
print(arc_output)
print()

print("H-ARC output:")
print(harc_output)
print()

print("VARC output:")
print(varc_output)
print()

print("ARC == H-ARC:", arc_output == harc_output)
print("ARC == VARC:", arc_output == varc_output)
print("H-ARC == VARC:", harc_output == varc_output)