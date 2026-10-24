#!/bin/bash
# Waits for Step 1 to finish, then runs Step 2 (KV extraction) and Step 3
# (LATTLE transfer, 10 pairs x 3 folds), then aggregates results.
cd "c:/Users/ikowsar/Desktop/Research/Lattle" || exit 1
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "WAITING for Step 1 to complete..."
while ! grep -q "ALL_STEP1_DONE" step1_driver.log 2>/dev/null; do
  sleep 60
done
echo "STEP1_COMPLETE at $(date +%H:%M:%S)"

# --- Step 2: extract K/V from every fine-tuned checkpoint ---
echo "STEP2_START at $(date +%H:%M:%S)"
$PY save_KV_weights.py > step2.log 2>&1
saved=$(grep -c "Saved K, V" step2.log)
echo "STEP2_DONE: $saved KV pairs extracted"
if [ "$saved" -lt 27 ]; then
  echo "STEP2_WARNING: expected 27, got $saved — see step2.log"
fi

# --- clear stale Step 3 state (old Optuna studies would be reused via
# load_if_exists=True, and old summaries would pollute aggregation) ---
rm -rf transfer_learning_results/optuna_studies
rm -rf transfer_learning_results/optuna_ckpt_results
echo "STEP3_START at $(date +%H:%M:%S) — 10 pairs x 3 folds"
$PY run_script-LATTLE.py > step3.log 2>&1
rc=$?
done_pairs=$(grep -c "Saved summary" step3.log)
echo "STEP3_DONE rc=$rc, $done_pairs/10 pairs completed at $(date +%H:%M:%S)"
if [ $rc -ne 0 ]; then
  echo "STEP3_ERROR — tail:"
  grep -E "Error|Traceback" step3.log | tail -5
fi

$PY aggregate_results.py > results_summary.md 2>&1
echo "AGGREGATED -> results_summary.md"
echo "ALL_DONE"
