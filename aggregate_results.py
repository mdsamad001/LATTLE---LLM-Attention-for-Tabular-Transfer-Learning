#!/usr/bin/env python3
"""Aggregate LATTLE results into paper-ready tables.

Reads:
  - step1_logs/{dataset}_{seed}.log         -> source LLM fine-tuning metrics
  - transfer_learning_results/optuna_ckpt_results/{src}_{tgt}/evaluation_summary_*.json
Writes:
  - results_step1_sources.csv
  - results_step3_transfer.csv
and prints both as markdown tables.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

STEP1_LOG_DIR = "step1_logs"
STEP3_GLOB = "./transfer_learning_results*/optuna_ckpt_results"

# index order must match dataset_pairs in run_script-LATTLE.py
PAIR_NAMES = [
    "credit-g_ilpd", "Cardiovascular-Disease-dataset_ilpd",
    "optdigits_vehicle", "mfeat-fourier_vehicle", "sick_cmc", "churn_cmc",
    "car_pc1", "steel-plate-fault_pc1",
    "steel-plate-fault_cylinder-bands", "seismic_bumps_cylinder-bands",
]


def parse_step1():
    rows = []
    for path in sorted(glob.glob(os.path.join(STEP1_LOG_DIR, "*.log"))):
        base = os.path.basename(path)[:-4]
        m = re.match(r"(.+)_(\d+)$", base)
        if not m:
            continue
        dataset, seed = m.group(1), int(m.group(2))
        vals = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                for key in ("AUC", "F1", "ACC", "Time"):
                    if s.startswith(key + ":"):
                        try:
                            vals[key] = float(s.split(":", 1)[1])
                        except ValueError:
                            pass
        if {"AUC", "F1", "ACC"} <= vals.keys():
            rows.append({"dataset": dataset, "seed": seed, **vals})
    return pd.DataFrame(rows)


def parse_step3():
    """Per-fold results.

    step3_logs/pair{P}_fold{F}.log is the authoritative record: each fold-run is
    a separate process that prints its own test metrics. The JSON summaries are
    written per pair and were historically overwritten when one worker ran
    several folds of the same pair, so the logs are the complete source.
    """
    rows = []
    pairs = PAIR_NAMES
    for path in sorted(glob.glob(os.path.join("step3_logs", "pair*_fold*.log"))):
        m = re.search(r"pair(\d+)_fold(\d+)\.log$", os.path.basename(path))
        if not m:
            continue
        pidx, fold = int(m.group(1)), int(m.group(2))
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        r = re.search(r"Fold\s+(\d+)\s+Results.*?AUC:\s*([\d.]+),\s*ACC:\s*([\d.]+),\s*F1:\s*([\d.]+)", text)
        if not r:
            continue
        rows.append({
            "pair": pairs[pidx] if pidx < len(pairs) else f"pair{pidx}",
            "fold": fold,
            "AUC": float(r.group(2)), "ACC": float(r.group(3)), "F1": float(r.group(4)),
        })
    if rows:
        return pd.DataFrame(rows)

    # fallback: the JSON summaries (used if logs are unavailable)
    for path in glob.glob(os.path.join(STEP3_GLOB, "*", "evaluation_summary_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        pair = os.path.basename(os.path.dirname(path))
        for fold_key, res in summary.get("fold_auc_scores", {}).items():
            rows.append({"pair": pair, "fold": int(fold_key.replace("fold_", "")),
                         "AUC": res["auc"], "ACC": res["acc"], "F1": res["f1"]})
    return pd.DataFrame(rows)


def summarize(df, group_col, metrics=("AUC", "F1", "ACC")):
    if df.empty:
        return df
    agg = df.groupby(group_col)[list(metrics)].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        mean, std = agg[(m, "mean")], agg[(m, "std")].fillna(0.0)
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean, std)]
    out["n"] = agg[(metrics[0], "count")]
    return out.reset_index()


def to_markdown(df, title):
    print(f"\n## {title}\n")
    if df.empty:
        print("_(no results yet)_")
        return
    print("| " + " | ".join(df.columns) + " |")
    print("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        print("| " + " | ".join(str(v) for v in r.values) + " |")


if __name__ == "__main__":
    s1 = parse_step1()
    if not s1.empty:
        s1.to_csv("results_step1_raw.csv", index=False)
        summary1 = summarize(s1, "dataset")
        summary1.to_csv("results_step1_sources.csv", index=False)
        to_markdown(summary1, "Step 1 — source LLM fine-tuning (mean ± std over seeds)")

    s3 = parse_step3()
    if not s3.empty:
        # Workers write to separate --tag dirs; if a (pair, fold) were ever run
        # twice it must not be counted twice in the mean.
        before = len(s3)
        s3 = s3.drop_duplicates(subset=["pair", "fold"], keep="first")
        if len(s3) < before:
            print(f"note: dropped {before - len(s3)} duplicate (pair, fold) result(s)")
        s3.to_csv("results_step3_raw.csv", index=False)
        summary3 = summarize(s3, "pair")
        summary3.to_csv("results_step3_transfer.csv", index=False)
        to_markdown(summary3, "Step 3 — LATTLE transfer (mean ± std over folds)")
        print(f"\nOverall mean transfer AUC: {s3.AUC.mean():.3f} "
              f"over {s3.pair.nunique()} pairs / {len(s3)} runs")
    else:
        print("\n(Step 3 results not present yet.)")
