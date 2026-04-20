from pathlib import Path
import json
import csv

# 全部400题的三个ARC output对比 - 拉表得出aligned_dataset.csv 在processed里面

def parse_harc_grid(grid_str: str):
    rows = [row for row in grid_str.strip().split("|") if row]
    grid = [[int(ch) for ch in row] for row in rows]
    return grid


# ---------- ARC ids ----------
arc_dir = Path("data/raw/ARC/evaluation")
arc_files = sorted(arc_dir.glob("*.json"))
arc_ids = sorted([f.stem for f in arc_files])

# ---------- H-ARC rows ----------
harc_file = Path("data/raw/H-ARC/summary_data.csv")
harc_map = {}

with harc_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["task_type"] == "evaluation":
            task_id = row["task_name"].replace(".json", "")
            if task_id not in harc_map:
                harc_map[task_id] = {
                    "solved": row["solved"],
                    "test_output_grid": row["test_output_grid"],
                }

# ---------- Build aligned rows ----------
rows = []

for task_id in arc_ids:
    # ARC official
    arc_file = Path(f"data/raw/ARC/evaluation/{task_id}.json")
    with arc_file.open("r", encoding="utf-8") as f:
        arc_data = json.load(f)
    arc_output = arc_data["test"][0]["output"]

    # H-ARC
    harc_output = None
    harc_solved = None
    if task_id in harc_map:
        harc_solved = harc_map[task_id]["solved"]
        harc_output = parse_harc_grid(harc_map[task_id]["test_output_grid"])

    # VARC
    varc_file = Path(f"data/raw/V-ARC/VARC_predictions/ARC-1_Unet/attempt_1/{task_id}_predictions.json")
    with varc_file.open("r", encoding="utf-8") as f:
        varc_data = json.load(f)
    varc_output = varc_data["0"][0]

    rows.append({
        "task_id": task_id,
        "arc_output": json.dumps(arc_output),
        "harc_output": json.dumps(harc_output),
        "varc_output": json.dumps(varc_output),
        "harc_solved": harc_solved,
        "arc_equals_harc": arc_output == harc_output,
        "arc_equals_varc": arc_output == varc_output,
        "harc_equals_varc": harc_output == varc_output,
    })

# ---------- Save ----------
output_file = Path("data/processed/aligned_dataset.csv")
with output_file.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task_id",
            "arc_output",
            "harc_output",
            "varc_output",
            "harc_solved",
            "arc_equals_harc",
            "arc_equals_varc",
            "harc_equals_varc",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_file}")
print("First 3 task_ids:", [row["task_id"] for row in rows[:3]])