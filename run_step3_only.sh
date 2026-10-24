#!/bin/bash
# Step 3 only, on the fixed pipeline. Step 1 checkpoints and K/V weights are
# reused from the existing run -- they are unaffected by the four fixes.
# usage: run_step3_only.sh [gpu_list] [n_folds] [n_trials]
cd "$(dirname "$0")" || exit 1
GPULIST=${1:-"0,1,2,3,4,5,6"}; NFOLDS=${2:-10}; NTRIALS=${3:-100}
IFS="," read -ra GPUS <<< "$GPULIST"; NW=${#GPUS[@]}
PY=.venv/bin/python
export PYTHONIOENCODING=utf-8
export LATTLE_SELECT_METRIC=${LATTLE_SELECT_METRIC:-val_loss}

echo "=== LATTLE Step 3 (FIXED pipeline): $NW workers on GPUs [$GPULIST], $NFOLDS folds, $NTRIALS trials"
echo "=== selection metric: $LATTLE_SELECT_METRIC"
date
echo "--- fixes in this run: model.train() per epoch | mode imputation | held-out selection on $LATTLE_SELECT_METRIC | refit w/o in-sample selection"

rm -rf transfer_learning_results_w* transfer_learning_results_smoke
echo "--- STEP3_START $(date +%H:%M:%S)"
for i in "${!GPUS[@]}"; do
  bash run_step3_parallel.sh "${GPUS[$i]}" "$i" "$NW" "$NFOLDS" "$NTRIALS" > "step3_w${i}.log" 2>&1 &
done
wait
n=$(cat transfer_learning_results_w*/optuna_ckpt_results/*/evaluation_summary_*.json 2>/dev/null | grep -c auc_mean)
echo "STEP3_DONE $(date +%H:%M:%S): $n pair summaries"
$PY aggregate_results.py > results_summary.md 2>&1
echo "AGGREGATED -> results_summary.md"
date
echo "ALL_DONE"
