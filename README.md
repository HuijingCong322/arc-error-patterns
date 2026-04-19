# Human vs. Model Error Patterns in ARC

**Group Members**: Jing Yu, Huijing Cong, Wanyi Zhou

## Project Overview
Comparing how humans and the VARC model make mistakes on ARC tasks.

## Setup

### 1. Clone this repo
git clone https://github.com/你的用户名/arc-error-patterns.git

### 2. Install dependencies
pip install huggingface_hub numpy pandas matplotlib

### 3. Download data

ARC original tasks:
git clone https://github.com/fchollet/ARC-AGI.git

VARC predictions:
huggingface-cli download VisionARC/VARC_predictions --repo-type dataset --local-dir ./VARC_predictions

H-ARC human data:
huggingface-cli download harc-dataset/H-ARC --repo-type dataset --local-dir ./HARC

## Folder Structure
arc-error-patterns/
├── data/
├── analysis/
├── notebooks/
└── results/

✅ VARC 完整资源汇总
论文
标题：ARC Is a Vision Problem!
arXiv 链接：https://arxiv.org/abs/2511.14761

官方代码仓库（最重要）
GitHub：https://github.com/lillian039/VARC

🎉 预测数据已公开！
官方仓库直接提供了预测数据，可以用以下命令下载：
hf download VisionARC/VARC_predictions --local-dir . --repo-type dataset
unzip VARC_predictions.zip
这些是 TTT（Test-Time Training）之后对每个任务的预测结果，包含 ARC-1 和 ARC-2 两个版本。
也就是说，你们不需要自己跑模型，直接下载现成的预测输出就可以用于错误分析！

模型简介（帮助你们理解数据）
VARC 把 ARC 任务重新定义为图像到图像的翻译问题（image-to-image translation），把 ARC 网格画在一个 64×64 的"画布"上，用标准视觉架构（ViT 或 U-Net）处理，通过测试时训练（TTT）泛化到未见任务。单个 18M 参数的 ViT 模型在 ARC-1 上达到 54.5% 准确率，集成后达到 60.4%，接近人类平均水平。
