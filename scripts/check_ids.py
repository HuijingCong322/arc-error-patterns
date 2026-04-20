from pathlib import Path
import csv

# 去掉后缀 只留track_id的数字

# ---------- ARC official ----------
arc_dir = Path("data/raw/ARC/evaluation")
arc_files = sorted(arc_dir.glob("*.json"))
arc_ids = {f.stem for f in arc_files}

print(f"ARC official evaluation tasks: {len(arc_ids)}")
print("Example ARC ids:", list(sorted(arc_ids))[:5])

# ---------- H-ARC ----------
harc_file = Path("data/raw/H-ARC/summary_data.csv")
harc_ids = set()

with harc_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["task_type"] == "evaluation":
            task_name = row["task_name"]
            task_id = task_name.replace(".json", "")
            harc_ids.add(task_id)

print(f"H-ARC evaluation tasks: {len(harc_ids)}")
print("Example H-ARC ids:", list(sorted(harc_ids))[:5])

# ---------- VARC ----------
varc_dir = Path("data/raw/V-ARC/VARC_predictions/ARC-1_Unet/attempt_1")
varc_files = sorted(varc_dir.glob("*_predictions.json"))
varc_ids = {f.name.replace("_predictions.json", "") for f in varc_files}

print(f"VARC ARC-1_Unet attempt_1 tasks: {len(varc_ids)}")
print("Example VARC ids:", list(sorted(varc_ids))[:5])

# ---------- overlap ----------
overlap_all = arc_ids & harc_ids & varc_ids
print(f"Overlap across ARC + H-ARC + VARC: {len(overlap_all)}")
print("Example overlap ids:", list(sorted(overlap_all))[:10])