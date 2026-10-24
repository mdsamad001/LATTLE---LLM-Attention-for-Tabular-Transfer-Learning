#!/bin/bash
# Step 3 worker: runs a slice of the (pair, fold) grid on one GPU.
# Splitting at fold granularity keeps all GPUs busy; each worker writes to its
# own --tag directory and aggregate_results.py merges them by pair.
# usage: run_step3_parallel.sh <gpu_id> <worker_idx> <n_workers> <n_folds> <n_trials>
GPU=$1; WID=$2; NW=$3; NFOLDS=${4:-10}; NTRIALS=${5:-15}
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
mkdir -p step3_logs

NPAIRS=10
i=0
for p in $(seq 0 $((NPAIRS-1))); do
  for f in $(seq 0 $((NFOLDS-1))); do
    if [ $((i % NW)) -eq "$WID" ]; then
      log="step3_logs/pair${p}_fold${f}.log"
      if grep -q "Saved summary" "$log" 2>/dev/null; then
        echo "[w$WID gpu$GPU] SKIP pair$p/fold$f"
      else
        echo "[w$WID gpu$GPU] START pair$p/fold$f $(date +%H:%M:%S)"
        $PY run_script-LATTLE.py --pair-index "$p" --fold-list "$f" \
            --trials "$NTRIALS" --tag "_w${WID}" > "$log" 2>&1
        if [ $? -eq 0 ]; then
          echo "[w$WID gpu$GPU] DONE pair$p/fold$f $(grep -oE 'AUC: [0-9.]+, ACC: [0-9.]+, F1: [0-9.]+' "$log" | tail -1)"
        else
          echo "[w$WID gpu$GPU] FAILED pair$p/fold$f"; grep -E "Error|Traceback" "$log" | tail -2
        fi
      fi
    fi
    i=$((i+1))
  done
done
echo "[w$WID gpu$GPU] WORKER_DONE"
