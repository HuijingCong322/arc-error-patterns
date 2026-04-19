import os
import subprocess
from huggingface_hub import snapshot_download
import urllib.request

print(">>> 开始下载数据...")

# 1. ARC 原始任务
print(">>> 下载 ARC 原始任务...")
if not os.path.exists("ARC-AGI"):
    subprocess.run(["git", "clone", "https://github.com/fchollet/ARC-AGI.git"])
    print("✓ ARC-AGI 下载完成")
else:
    print("✓ ARC-AGI 已存在，跳过")

# 2. VARC 预测数据
print("\n>>> 下载 VARC 预测数据...")
if not os.path.exists("VARC_predictions"):
    snapshot_download(
        repo_id="VisionARC/VARC_predictions",
        repo_type="dataset",
        local_dir="./VARC_predictions"
    )
    print("✓ VARC_predictions 下载完成")
else:
    print("✓ VARC_predictions 已存在，跳过")

# 3. H-ARC 人类数据（从OSF下载）
print("\n>>> 下载 H-ARC 数据...")
os.makedirs("HARC", exist_ok=True)

files = {
    "arc_responses.csv": "https://osf.io/download/s3xqd/",
    "arc_action_traces.csv": "https://osf.io/download/yfnmv/",
    "arc_nl_descriptions.csv": "https://osf.io/download/3rjet/",
}

for filename, url in files.items():
    filepath = f"HARC/{filename}"
    if not os.path.exists(filepath):
        print(f"  下载 {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✓ {filename} 完成")
    else:
        print(f"  ✓ {filename} 已存在，跳过")

print("\n=== 所有数据下载完成！===")