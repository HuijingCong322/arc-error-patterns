from pathlib import Path
import json
import csv

# 检查三个arc数据到底长什么样 发现V-ARC的不同

task_id = "0a1d4ef5"

# ---------- ARC official ----------
arc_file = Path(f"data/raw/ARC/evaluation/{task_id}.json")
with arc_file.open("r", encoding="utf-8") as f:
    arc_data = json.load(f)

arc_test_output = arc_data["test"][0]["output"]

print("ARC official task id:", task_id)
print("ARC official test output:")
print(arc_test_output)
print()

# ---------- H-ARC ----------
harc_file = Path("data/raw/H-ARC/summary_data.csv")
found_harc = False

with harc_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["task_type"] == "evaluation" and row["task_name"] == f"{task_id}.json":
            print("Found one H-ARC row:")
            print("task_name:", row["task_name"])
            print("solved:", row["solved"])
            print("test_output_grid:", row["test_output_grid"][:200])
            found_harc = True
            break

if not found_harc:
    print("No H-ARC row found.")
print()

# ---------- VARC ----------
varc_file = Path(f"data/raw/V-ARC/VARC_predictions/ARC-1_Unet/attempt_1/{task_id}_predictions.json")
with varc_file.open("r", encoding="utf-8") as f:
    varc_data = json.load(f)

print("VARC file keys:")
print(varc_data.keys())
print("VARC raw content preview:")
print(str(varc_data)[:500])