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

## 数据处理方法

### ARC Ground Truth
直接读取 `ARC-AGI/data/evaluation/` 下 400 个 JSON 文件，每个任务取 `test[i]["output"]` 作为标准答案。

### VARC 预测（`analysis/load_data.py: load_varc_predictions`）
- 使用模型：`ARC-1_ViT`（ViT 架构，ARC-1 evaluation set）
- 读取 `attempt_0` 到 `attempt_3` 共 4 次独立预测
- 对每道题的所有预测做 **majority vote**（取出现次数最多的网格）作为最终预测
- 这是 VARC 论文的官方评估方式

### H-ARC 人类数据（`analysis/load_data.py: load_harc_summary`）
- 数据来源：`HARC/data/summary_data.csv`，过滤 `task_type == "evaluation"`
- 共 946 名参与者，400 道题，7820 条记录（平均每题 10.3 人）
- 每人最多有 3 次尝试机会，取**每位参与者的最后一次提交**作为其最终答案
- 每道题的人类准确率 = 答对的参与者数 / 参与该题的参与者总数
- 这样每道题有约 10 个独立数据点，而非单人代表

### 错误分类（`analysis/error_analysis.py`）
对每个预测网格与标准答案比较，分为四类：
| 类型 | 定义 |
|------|------|
| `correct` | 完全匹配 |
| `wrong_size` | 输出尺寸（行数或列数）不一致 |
| `close_miss` | 尺寸正确，cell accuracy ≥ 80% |
| `wrong_content` | 尺寸正确，cell accuracy < 80% |

### 任务级对比（`task_level_summary`）
- VARC：所有 test example 全部正确才算该任务正确（ARC 官方评分标准）
- Human：以参与者中答对比例 > 50% 为该任务"人类可解"

## 代码结构

analysis/load_data.py — 三个数据集的加载函数
- `load_arc_ground_truth()` → 400个任务的标准答案
- `load_varc_predictions()` → 跨4次attempt合并，majority vote
- `load_harc_summary()` → 加载 summary_data.csv，过滤 evaluation 任务
- `load_harc_incorrect_submissions()` → 加载聚合错误数据

analysis/error_analysis.py — 错误分类与对比
- 四类错误：correct / wrong_size / close_miss / wrong_content
- `compute_varc_errors()` 和 `compute_human_errors()`（取最后一次尝试）
- `task_level_summary()` → 每道题的人类准确率 vs. VARC 对错

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
