#!/bin/bash

# 로그를 저장할 디렉토리 생성
mkdir -p logs

echo "========================================================"
echo "[DEBUG] Starting 4 Experiments Sequentially"
echo "========================================================"

# 1. Baseline Experiment
echo ""
echo "[1/4] Running Baseline Experiment..."
uv run python -u run_finetuning_baseline.py 2>&1 | tee -a logs/log_baseline.txt

# 2. Deletion Experiment
echo ""
echo "[2/4] Running Deletion Experiment..."
uv run python -u run_finetuning_deletion.py 2>&1 | tee -a logs/log_deletion.txt

# 3. Context Insertion Experiment
echo ""
echo "[3/4] Running Context Insertion (Que) Experiment..."
uv run python -u run_finetuning_insertion_que.py 2>&1 | tee -a logs/log_insertion_que.txt

# 4. Answer Insertion Experiment
echo ""
echo "[4/4] Running Answer Insertion (Ans) Experiment..."
uv run python -u run_finetuning_insertion_ans.py 2>&1 | tee -a logs/log_insertion_ans.txt

echo ""
echo "========================================================"
echo "[DEBUG] All Experiments Completed."
echo "========================================================"