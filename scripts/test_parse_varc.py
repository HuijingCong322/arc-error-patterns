from pathlib import Path
import json

# 找出V-ARC里面哪里用来和其他俩比较的

task_id = "0a1d4ef5"

varc_file = Path(f"data/raw/V-ARC/VARC_predictions/ARC-1_Unet/attempt_1/{task_id}_predictions.json")

with varc_file.open("r", encoding="utf-8") as f:
    varc_data = json.load(f)

print("Full keys:", varc_data.keys())
print()

preds = varc_data["0"]
print("Type of varc_data['0']:", type(preds))
print("Number of predictions inside:", len(preds))
print()

print("First prediction:")
print(preds[0])
print()

if len(preds) > 1:
    print("Second prediction:")
    print(preds[1])