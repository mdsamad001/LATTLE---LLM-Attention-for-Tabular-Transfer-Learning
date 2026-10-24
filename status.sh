#!/bin/bash
# One-shot status of the LATTLE run. Usage: bash status.sh
cd "$(dirname "$0")" || exit 1
echo "=== phase ==="; grep -E "STEP[0-9]_(START|DONE)|PREWARM_OK|AGGREGATED|ALL_DONE" master.log 2>/dev/null | tail -6
echo "=== progress ==="
echo "  step1: $(cat step1_w*.log 2>/dev/null | grep -c '] DONE')/90 done, $(cat step1_w*.log 2>/dev/null | grep -c '] FAILED') failed"
echo "  step3: $(cat step3_w*.log 2>/dev/null | grep -c '] DONE')/100 done, $(cat step3_w*.log 2>/dev/null | grep -c '] FAILED') failed"
echo "=== alive? ==="
n=$(pgrep -u "$USER" -f "Finetune-LLM|run_script-LATTLE" | wc -l)
echo "  $n worker processes; driver $(pgrep -u "$USER" -f run_all_parallel.sh >/dev/null && echo UP || echo DOWN)"
echo "=== gpus ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== last 5 completions ==="; cat step1_w*.log step3_w*.log 2>/dev/null | grep '] DONE' | tail -5
