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
运行download_data.py，下载ARC 原始任务和 VARC 预测数据
但需手动下载 H-ARC（只需做一次）
第一步：打开 OSF 页面
浏览器打开：https://osf.io/bh8yq
第二步：下载 zip 文件
点击页面上的 "OSF Storage" → 找到 osfstorage-archive.zip → 点击下载
对我们的项目来说，只需要：
HARC/data/incorrect_submissions.csv — 每道题上所有人的错误答案汇总，这就是"人类错误数据"

### 4.运行notebook文件夹下的compare_errors.ipynb
我加了一点结果解释,运行结果在result文件夹

## Folder Structure
arc-error-patterns/
├── data/
├── analysis/
├── notebooks/
└── results/

## 代码结构

analysis/load_data.py — 三个数据集的加载函数
- load_arc_ground_truth() → 400个任务的标准答案
- load_varc_predictions() → 自动 majority vote，返回最终预测
- load_harc_responses() → 加载 H-ARC CSV，自动尝试常见文件名

analysis/error_analysis.py — 错误分类与对比
- 四类错误：correct / wrong_size / close_miss / wrong_content
- compute_varc_errors() 和 compute_human_errors()（只取最后一次尝试）
- task_level_summary() → 每道题的人类准确率 vs. VARC 对错

notebooks/compare_errors.ipynb — 分析图
- 错误类型分布柱状图
- 每题准确率散点图（四象限）
- 答错时的 cell accuracy 分布

## 资源

### VARC
- 论文：[ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761)
- 代码：https://github.com/lillian039/VARC
- 预测数据：`hf download VisionARC/VARC_predictions --local-dir . --repo-type dataset`

### H-ARC
- 网站：https://exps.gureckislab.org/e/assumption-fast-natural/#/
- 数据：https://osf.io/bh8yq
