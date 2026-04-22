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

## 语义错误标注（下一步可能要做的）

对于人类和 VARC 都答错的任务（`both_wrong`，共 73 道），我们通过人工查看任务图形来分析具体的错误原因，并打上语义标签。

### 错误类型定义

| 标签 | 含义 |
|------|------|
| `wrong_position` | 找到了正确的对象或模式，但放错了位置 |
| `wrong_structure` | 颜色对了，但整体结构或形状错了 |
| `partial_rule` | 部分理解了规则，但没有完整应用 |
| `near_miss` | 非常接近正确答案，只差 1-2 个格子或一步变换 |
| `wrong_rule` | 完全用了错误的规则 |
| `no_pattern` | 输出看起来像随机填写，没有体现对规则的理解 |

### 如何查看任务

打开 `notebooks/compare_errors.ipynb`，滚动到 **Section 8**。

**第一步** — 运行 `both_wrong_ids` 那个 cell，获取 73 道题的 task_id 列表。

**第二步** — 调用 `show_task()` 可视化任意任务：
```python
# 显示任务 + VARC 预测（默认同时显示最多6个人类错误提交）
show_task("ad7e01d0", varc_pred=varc_predictions["ad7e01d0"][0])

# 不显示人类提交（只看任务结构和 VARC）
show_task("ad7e01d0", varc_pred=varc_predictions["ad7e01d0"][0], max_humans=0)

# 显示更多人类提交
show_task("ad7e01d0", varc_pred=varc_predictions["ad7e01d0"][0], max_humans=10)
```

可视化说明：

- **上两排**：训练示例（输入/输出）+ 测试输入 + Ground Truth + VARC预测（绿色标题=对，红色=错）
- **下两排**：该任务中人类最后一次提交的错误答案（最多 `max_humans` 个，红色标题）

**第三步** — 看完每道题后，在 Section 8 底部的标注表格里填写错误标签和备注。

## 资源

### VARC
- 论文：[ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761)
- 代码：https://github.com/lillian039/VARC
- 预测数据：`hf download VisionARC/VARC_predictions --local-dir . --repo-type dataset`

### H-ARC
- 网站：https://exps.gureckislab.org/e/assumption-fast-natural/#/
- 数据：https://osf.io/bh8yq
