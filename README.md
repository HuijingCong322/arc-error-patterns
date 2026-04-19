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
