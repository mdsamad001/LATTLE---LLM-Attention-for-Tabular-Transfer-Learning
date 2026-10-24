#!/bin/bash
# Step 1 driver: fine-tune DistilGPT2 on every source dataset x seed.
# Resumable: skips (dataset, seed) combos whose base checkpoint already exists.
cd "c:/Users/ikowsar/Desktop/Research/Lattle" || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p step1_logs

DATASETS=(credit-g car sick optdigits mfeat-fourier churn steel-plate-fault seismic_bumps Cardiovascular-Disease-dataset)
SEEDS=(0 1 2)

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt="./llm_finetuning_results/best_models/distilgpt-gftt-last_1_${ds}_${seed}/base/config.json"
    if [ -f "$ckpt" ]; then
      echo "SKIP $ds seed $seed (checkpoint exists)"
      continue
    fi
    log="step1_logs/${ds}_${seed}.log"
    echo "START $ds seed $seed at $(date +%H:%M:%S)"
    $PY -m LLM_finetune_module.Finetune-LLM \
      --dataset "$ds" --seed "$seed" --batch 16 --prompt_template 1 \
      --name distilgpt-gftt-last > "$log" 2>&1
    if [ $? -eq 0 ]; then
      metrics=$(grep -E '^(AUC|F1|ACC|Time):' "$log" | tr '\n' ' ')
      echo "DONE $ds seed $seed -> $metrics"
    else
      echo "FAILED $ds seed $seed (see $log)"
      tail -3 "$log"
    fi
  done
done
echo "ALL_STEP1_DONE"
