#!/bin/bash
# Master driver: full LATTLE suite across multiple GPUs.
#   Step 1 (fine-tune) -> Step 2 (K/V extract) -> Step 3 (transfer) -> aggregate
# usage: run_all_parallel.sh [gpu_list] [n_seeds] [n_trials]
#   e.g. run_all_parallel.sh "1,2,3,4,5,6" 10 15
cd "$(dirname "$0")" || exit 1
GPULIST=${1:-"1,2,3,4,5,6"}
NSEEDS=${2:-10}
NTRIALS=${3:-15}
IFS=',' read -ra GPUS <<< "$GPULIST"
NW=${#GPUS[@]}
PY=.venv/bin/python
export PYTHONIOENCODING=utf-8

echo "=== LATTLE full run: $NW workers on GPUs [$GPULIST], $NSEEDS seeds, $NTRIALS trials ==="
date

echo "--- PREWARM"
NFOLDS=$NSEEDS $PY prewarm.py > prewarm.log 2>&1
grep -q PREWARM_DONE prewarm.log && echo "PREWARM_OK" || { echo "PREWARM_FAILED"; tail -5 prewarm.log; exit 1; }

echo "--- STEP1_START $(date +%H:%M:%S)"
for i in "${!GPUS[@]}"; do
  bash run_step1_parallel.sh "${GPUS[$i]}" "$i" "$NW" "$NSEEDS" > "step1_w${i}.log" 2>&1 &
done
wait
echo "STEP1_DONE $(date +%H:%M:%S): $(grep -h -c '] DONE' step1_w*.log | paste -sd+ | bc) ok, $(grep -h -c '] FAILED' step1_w*.log | paste -sd+ | bc) failed"

echo "--- STEP2_START $(date +%H:%M:%S)"
sed -i "s/seeds = list(range([0-9]*))/seeds = list(range($NSEEDS))/" save_KV_weights.py
$PY save_KV_weights.py > step2.log 2>&1
echo "STEP2_DONE: $(grep -c 'Saved K, V' step2.log) KV pairs"

echo "--- STEP3_START $(date +%H:%M:%S)"
rm -rf transfer_learning_results_w*
for i in "${!GPUS[@]}"; do
  bash run_step3_parallel.sh "${GPUS[$i]}" "$i" "$NW" "$NSEEDS" "$NTRIALS" > "step3_w${i}.log" 2>&1 &
done
wait
echo "STEP3_DONE $(date +%H:%M:%S): $(cat transfer_learning_results_w*/optuna_ckpt_results/*/evaluation_summary_*.json 2>/dev/null | grep -c auc_mean) pair summaries"

$PY aggregate_results.py > results_summary.md 2>&1
echo "AGGREGATED -> results_summary.md"
date
echo "ALL_DONE"
