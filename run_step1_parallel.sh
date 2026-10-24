#!/bin/bash
# Step 1 worker: fine-tunes a slice of the (dataset, seed) job grid on one GPU.
# usage: run_step1_parallel.sh <gpu_id> <worker_idx> <n_workers> <n_seeds>
GPU=$1; WID=$2; NW=$3; NSEEDS=${4:-10}
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
mkdir -p step1_logs

DATASETS=(credit-g car sick optdigits mfeat-fourier churn steel-plate-fault seismic_bumps Cardiovascular-Disease-dataset)

i=0
for ds in "${DATASETS[@]}"; do
  for seed in $(seq 0 $((NSEEDS-1))); do
    if [ $((i % NW)) -eq "$WID" ]; then
      ckpt="./llm_finetuning_results/best_models/distilgpt-gftt-last_1_${ds}_${seed}/base/config.json"
      if [ -f "$ckpt" ]; then
        echo "[w$WID gpu$GPU] SKIP $ds/$seed"
      else
        echo "[w$WID gpu$GPU] START $ds/$seed $(date +%H:%M:%S)"
        $PY -m LLM_finetune_module.Finetune-LLM --dataset "$ds" --seed "$seed" \
            --batch 16 --prompt_template 1 --name distilgpt-gftt-last \
            > "step1_logs/${ds}_${seed}.log" 2>&1
        if [ $? -eq 0 ]; then
          echo "[w$WID gpu$GPU] DONE $ds/$seed -> $(grep -E '^(AUC|F1|ACC):' step1_logs/${ds}_${seed}.log | tr '\n' ' ')"
        else
          echo "[w$WID gpu$GPU] FAILED $ds/$seed"; tail -3 "step1_logs/${ds}_${seed}.log"
        fi
      fi
    fi
    i=$((i+1))
  done
done
echo "[w$WID gpu$GPU] WORKER_DONE"
