#!/usr/bin/env python3
"""TransTab baseline under the EXACT LATTLE protocol (apples-to-apples).

Identical to run_script-LATTLE.py in every respect - same cached splits, same
folds, same Optuna space and budget, same val-AUC checkpoint selection, same
Stage-2 refit on train+val for the learned epoch budget, same test evaluation,
same (fixed) trainer. The ONLY difference is what initialises the gFTT:

  --mode notransfer : nothing. Plain gFTT trained on the target. Isolates the
                      entire contribution of the K/V transplant.
  --mode transfer   : TransTab-style source->target transfer. A gFTT is trained
                      on the SOURCE table (Stage 0), then loaded and fine-tuned
                      on the target. This is the TransTab baseline of Table 2.

modeling_lattle.py is a fork of TransTab modeling_transtab.py (same 13 classes),
so `notransfer` IS TransTab architecture - only apply_ca() differs.
"""
import os
import json
import torch
import optuna
import pickle


import numpy as np
import pandas as pd

from transfer_learning_module import lattle
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--pair-index', type=int, default=None,
                 help='run only this index of dataset_pairs (for parallel workers)')
_ap.add_argument('--folds', type=int, default=3)
_ap.add_argument('--fold-list', type=str, default=None,
                 help='comma-separated folds to run (overrides --folds); lets '
                      'workers split folds of one pair across GPUs')
_ap.add_argument('--trials', type=int, default=15)
_ap.add_argument('--tag', type=str, default='',
                 help='suffix for output dirs so parallel workers do not collide')
_ap.add_argument('--mode', choices=['notransfer', 'transfer'], default='transfer',
                 help="notransfer: plain gFTT on the target (isolates the transplant). "
                      "transfer: TransTab source->target transfer (Table 2 baseline).")
# NOTE: no --arch switch. build_classifier's num_layer argument is ignored by
# TransTabModel (verified: num_layer=2 and num_layer=5 both build 5 attention
# layers), so a "TransTab-default geometry" option could not be honoured. The
# geometry is therefore matched to LATTLE, which is also the setting that
# isolates the K/V transplant as the only difference.
_args = _ap.parse_args()

WORK = f'./transtab_results_{_args.mode}{_args.tag}'
# 'matched' keeps LATTLE geometry so the comparison isolates the transplant;
# 'transtab' uses the TransTab paper's own defaults.
ARCH = dict(num_layer=5, ffn_dim=2048, num_attention_head=8, hidden_dim=768)
# Stage 0 (source pre-training) hyperparameters: fixed, not tuned - mirroring how
# LATTLE fixes its Step-1 LLM hyperparameters rather than searching them.
SRC_CKPT_ROOT = './transtab_source_ckpt'
# (source, fold) -> (checkpoint dir, source class count). Filled once per fold
# BEFORE the Optuna study starts: doing it lazily inside objective() would make
# the first trial of each fold consume 150 epochs of RNG that later trials do
# not, so trial 1 would not be comparable with trials 2..N.
SOURCE_CKPT = {}
SRC_HP = dict(lr=1e-4, batch_size=64, weight_decay=1e-5,
              hidden_dropout_prob=0.1, warmup_ratio=0.1)



# === Configuration ===
# Source names match the Step-1/Step-2 dataset names so KV weights resolve
# under KV_DIR/{source}_{fold}.
dataset_pairs = [
    ("credit-g", "ilpd"),
    ("Cardiovascular-Disease-dataset", "ilpd"),
    ("optdigits", "vehicle"),
    ("mfeat-fourier", "vehicle"),
    ("sick", "cmc"),
    ("churn", "cmc"),
    ("car", "pc1"),
    ("steel-plate-fault", "pc1"),
    ("steel-plate-fault", "cylinder-bands"),
    ("seismic_bumps", "cylinder-bands"),
]

num_class_target_map = {
    "ilpd": 2,
    "vehicle": 4,
    "cmc": 3,
    "pc1": 2,
    "cylinder-bands": 2
}

folds = ([int(x) for x in _args.fold_list.split(',')]
         if _args.fold_list else list(range(_args.folds)))
KV_DIR = "./transfer_learning_module/gpt_FTTs_GPT-last_layer_KV_weights"

# Model-selection metric for Stage 1 (tuning), scored on the held-out val split.
# Default 'val_loss' is lower-is-better and relies on the direction-aware comparison
# in trainer.train(). Set LATTLE_SELECT_METRIC=auc to select on val AUC instead.
# Reported test metrics (AUC/ACC/F1) are computed the same way either way.
SELECT_METRIC = os.environ.get('LATTLE_SELECT_METRIC', 'val_loss')
SELECT_LESS_IS_BETTER = (SELECT_METRIC == 'val_loss')
# Fixed epoch budget for Stage 1 (paper Sec 4.1 = 150). Overridable only so the
# plumbing can be smoke-tested cheaply; production runs must leave it at 150.
MAX_EPOCH = int(os.environ.get('LATTLE_MAX_EPOCH', '150'))
RESULTS_ROOT = f"{WORK}/optuna_ckpt_results"




def get_cache_path(dataset_name, seed=None):
    """Create unique cache path for dataset and seed."""
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    return os.path.join("cached_datasets", f"{dataset_name}{seed_suffix}.pkl")

def load_cached_data(dataset_name, seed=None):
    """
    Loads dataset from cache if available; otherwise loads from OpenML and caches it.
    Returns: trainset, valset, testset, cat_cols, num_cols, bin_cols
    """
    os.makedirs("cached_datasets", exist_ok=True)
    cache_path = get_cache_path(dataset_name, seed)

    # Check if cache exists
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"✅ Loaded cached dataset: {dataset_name} (seed={seed})")
            return pickle.load(f)

    # Else, load from OpenML
    print(f"⬇️  Downloading dataset: {dataset_name} from OpenML...")
    if seed is not None:
        result = lattle.load_data([dataset_name], seed=seed)
    else:
        result = lattle.load_data([dataset_name])

    # Cache the result
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
        print(f"💾 Cached dataset: {dataset_name} at {cache_path}")

    return result


def set_seed(seed: int):
    lattle.random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def pretrain_source(source, fold):
    """Stage 0: train a gFTT on the SOURCE table and cache the checkpoint.

    Runs once per (source, fold) - mirrors LATTLE Step 1, which fine-tunes the
    source LLM once per (source, seed) rather than inside the Optuna loop.
    Selection uses the source's own held-out val split and the same metric.
    """
    ckpt = os.path.join(SRC_CKPT_ROOT, f"{source}_{fold}")
    marker = os.path.join(ckpt, 'n_class.txt')
    if os.path.exists(marker):
        return ckpt, int(open(marker).read().strip())
    set_seed(fold)
    _, trainset, valset, _, cat_cols, num_cols, bin_cols = load_cached_data(source, seed=fold)
    Xs, ys = trainset[0]
    Xsv, ysv = valset[0]
    n_src = int(pd.Series(ys).nunique())
    print(f"   [stage0] pre-training TransTab on source {source} fold {fold} ({n_src} classes)")
    m = lattle.build_classifier(cat_cols, num_cols, bin_cols,
                                num_class=n_src, p=1,
                                hidden_dropout_prob=SRC_HP['hidden_dropout_prob'], **ARCH)
    lattle.train(m, (Xs, ys), (Xsv, ysv),
                 num_epoch=MAX_EPOCH,
                 eval_metric=SELECT_METRIC, eval_less_is_better=SELECT_LESS_IS_BETTER,
                 output_dir=f'{WORK}/stage0', seed=fold,
                 batch_size=SRC_HP['batch_size'], lr=SRC_HP['lr'],
                 weight_decay=SRC_HP['weight_decay'],
                 hidden_dropout_prob=SRC_HP['hidden_dropout_prob'],
                 num_class=n_src, save_best=True,
                 exp=f'{source}_src_fold_{fold}', warmup_ratio=SRC_HP['warmup_ratio'])
    # save_best=True so the trainer restores the val-selected epoch before we
    # serialise. With save_best=False the trainer neither writes nor restores a
    # checkpoint, so m would be the last (150th) epoch while LATTLE's Step-1
    # source model IS best-val selected - an asymmetry favouring LATTLE.
    os.makedirs(ckpt, exist_ok=True)
    m.save(ckpt)
    with open(os.path.join(ckpt, 'n_class.txt'), 'w') as fh:
        fh.write(str(n_src))
    print(f"   [stage0] saved source checkpoint ({n_src} classes) -> {ckpt}")
    return ckpt, n_src


def _match_lattle_weight_decay(model):
    """Exempt gFTT l0's in_proj_weight from weight decay, as LATTLE does.

    In LATTLE this exemption is a side effect of protecting the transplanted
    K/V, which share one in_proj_weight tensor with Q - so l0's query is also
    exempted. Reproducing it here keeps the optimizer grouping identical; no
    gradient hook is registered, so the baseline's K/V stay trainable, which is
    the intended difference between the two methods.
    """
    layers = model.gftt_attention_layers()
    if layers:
        model.frozen_kv_params = [layers[0].self_attn.in_proj_weight]
    return model


def init_model(source, target, num_class_target, fold, dropout,
               cat_cols, num_cols, bin_cols):
    """Build the target gFTT exactly as LATTLE does, then apply the baseline's
    initialisation in place of the K/V transplant.

    The head needs care. TransTabModel.update() only rebuilds the classifier
    when the requested class count DIFFERS from self.num_class, and load() calls
    load_state_dict(strict=False), which suppresses missing/unexpected keys but
    still raises on a size mismatch. So the model must be built with the SOURCE
    class count (so the source head loads cleanly) and only then switched to the
    target head. Building with the target count instead makes update() a no-op:
    where the counts coincide the source head is silently inherited, and where
    they differ load() raises RuntimeError.
    """
    if _args.mode != 'transfer':
        return _match_lattle_weight_decay(
            lattle.build_classifier(cat_cols, num_cols, bin_cols,
                                    num_class=num_class_target, p=1,
                                    hidden_dropout_prob=dropout, **ARCH))

    ckpt, n_src = SOURCE_CKPT[(source, fold)]
    model = lattle.build_classifier(cat_cols, num_cols, bin_cols,
                                    num_class=n_src, p=1,
                                    hidden_dropout_prob=dropout, **ARCH)
    model.load(ckpt)                                       # source weights
    model.update({'cat': cat_cols, 'num': num_cols, 'bin': bin_cols})
    # Re-initialise the head unconditionally - including when n_src happens to
    # equal the target count - so the target always starts from a fresh
    # classifier, exactly as LATTLE does. The sentinel forces the guard in
    # _adapt_to_new_num_class to fire.
    model.num_class = -1
    model.update({'num_class': num_class_target})
    expected = 1 if num_class_target <= 2 else num_class_target
    assert model.clf.fc.out_features == expected, (
        f"head width {model.clf.fc.out_features} != expected {expected}")
    return _match_lattle_weight_decay(model)


def objective(trial, source, target, num_class_target, fold):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dropout_prob = trial.suggest_categorical("hidden_dropout_prob", [0.0, 0.1, 0.2, 0.3, 0.4])
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.01, 0.05, 0.1])


    

    set_seed(fold)
    # _, trainset, valset, _, cat_cols, num_cols, bin_cols = transtab.load_data([target])
    _, trainset, valset, _, cat_cols, num_cols, bin_cols = load_cached_data(target, seed=fold)
    X_train, y_train = trainset[0]
    X_val, y_val = valset[0]

    # Dropout must be set at BUILD time (train() cannot retrofit nn.Dropout p).
    model = init_model(source, target, num_class_target, fold,
                       hidden_dropout_prob, cat_cols, num_cols, bin_cols)

    # Stage 1 (tuning): train on TRAIN only, select on the held-out VAL split.
    # No checkpoint is written (save_best=False); train() returns the best score
    # and the epoch it occurred at, so scoring a trial at its peak costs no I/O.
    res = lattle.train(
        model,
        (X_train, y_train), (X_val, y_val),
        num_epoch=MAX_EPOCH,   # paper Sec 4.1; no early stopping by design
        eval_metric=SELECT_METRIC,
        eval_less_is_better=SELECT_LESS_IS_BETTER,
        output_dir=f'{WORK}/trials',
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dropout_prob=hidden_dropout_prob,
        num_class=num_class_target,
        save_best=False,
        exp=f'{source}_{target}_fold_{fold}_optuna',
        warmup_ratio=warmup_ratio,
    )

    # Record the epoch budget for Stage 2; Optuna only ranks on the score.
    trial.set_user_attr('best_epoch', int(res['best_epoch']))
    trial.set_user_attr('select_metric', SELECT_METRIC)
    return float(res['best_eval_res'])


def evaluate_final(source, target, num_class_target, best_params, fold, best_epoch):
    # 1) fix the seed
    set_seed(fold)

    # 2) load all three splits
    # _, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data([target], seed = fold)
    _, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_cached_data(target, seed = fold)
    X_tr, y_tr = trainset[0]
    X_val, y_val = valset[0]
    X_te, y_te = testset[0]

    # 3) wrap as pandas (so .iloc will work)
    X_tr_df   = pd.DataFrame(X_tr)
    y_tr_s    = pd.Series(y_tr)
    X_val_df  = pd.DataFrame(X_val)
    y_val_s   = pd.Series(y_val)
    X_te_df   = pd.DataFrame(X_te)
    y_te_s    = pd.Series(y_te)

    # 4) merge train + val
    X_train_val = pd.concat([X_tr_df, X_val_df], ignore_index=True)
    y_train_val = pd.concat([y_tr_s,  y_val_s],  ignore_index=True)

    # 5) rebuild with the baseline's initialisation (no K/V transplant)
    model = init_model(source, target, num_class_target, fold,
                       best_params['hidden_dropout_prob'], cat_cols, num_cols, bin_cols)
    # 6) Stage 2 (refit): train on train+val for exactly the epoch budget learned
    # in Stage 1. No checkpoint selection here — val is now inside the training
    # data, so selecting on it would be in-sample. No validation set is passed,
    # so nothing but the test predict below ever sees held-out data.
    lattle.train(
        model,
        (X_train_val, y_train_val),
        None,
        num_epoch=max(1, int(best_epoch) + 1),   # best_epoch is 0-indexed
        eval_metric=SELECT_METRIC,
        eval_less_is_better=SELECT_LESS_IS_BETTER,
        output_dir=WORK,
        seed=fold,
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        hidden_dropout_prob=best_params['hidden_dropout_prob'],
        num_class=num_class_target,
        save_best=False,
        exp=f'{source}_{target}_final_fold_{fold}',
        warmup_ratio=best_params['warmup_ratio'],
    )

    # 7) predict & compute metrics on the *original* NumPy test arrays
    y_pred = lattle.predict(model, X_te, y_te)
    if num_class_target == 2:
        y_score    = y_pred[:,1] if y_pred.ndim==2 else y_pred
        y_pred_cls = (y_score > 0.5).astype(int)
        auc  = roc_auc_score(y_te,    y_score)
        acc  = accuracy_score(y_te,   y_pred_cls)
        f1   = f1_score(y_te,         y_pred_cls, average="binary")
    else:
        y_pred_cls = np.argmax(y_pred, axis=1)
        auc  = roc_auc_score(y_te,    y_pred,   multi_class="ovr", average="macro")
        acc  = accuracy_score(y_te,   y_pred_cls)
        f1   = f1_score(y_te,         y_pred_cls, average="macro")

    return {"auc": round(auc,3), "acc": round(acc,3), "f1": round(f1,3)}



if __name__ == '__main__':
    if _args.pair_index is not None:
        dataset_pairs = [dataset_pairs[_args.pair_index]]
    for source, target in dataset_pairs:
        print(f"\n🔁 Running for: {source} → {target}")
        num_class_target = num_class_target_map[target]

        all_fold_results = {}
        all_best_params = {}
        auc_list, acc_list, f1_list = [], [], []

        for fold in folds:
            print(f"\n🔧 Tuning Fold {fold} for: {source} → {target}")
            if _args.mode == 'transfer':
                # Hoisted out of objective() so every trial sees the same RNG
                # stream; cached on disk so it costs nothing after the first run.
                SOURCE_CKPT[(source, fold)] = pretrain_source(source, fold)
                set_seed(fold)
            # study = optuna.create_study(direction='maximize')

            
            # Optional: create a unique study name per fold
            study_name = f"{source}_{target}_fold_{fold}"
            
            # Create full directory
            optuna_dir = f"{WORK}/optuna_studies"
            os.makedirs(optuna_dir, exist_ok=True)
            
            # Save DB in that location
            storage_path = f"sqlite:///{optuna_dir}/{study_name}.db"

            study = optuna.create_study(
                study_name=study_name,
                direction='minimize' if SELECT_LESS_IS_BETTER else 'maximize',
                storage=storage_path,
                load_if_exists=True  # allows continuing from past runs
            )

            study.optimize(
                lambda tr: objective(tr, source, target, num_class_target, fold),
                n_trials=_args.trials
            )

            best_trial = study.best_trial
            best_params = best_trial.params
            best_epoch = best_trial.user_attrs['best_epoch']
            all_best_params[f'fold_{fold}'] = dict(best_params, best_epoch=best_epoch)
            print(f"   best {SELECT_METRIC}={best_trial.value:.6f} at epoch {best_epoch}")

            result = evaluate_final(source, target, num_class_target, best_params, fold, best_epoch)
            all_fold_results[f'fold_{fold}'] = result
            auc_list.append(result['auc'])
            acc_list.append(result['acc'])
            f1_list.append(result['f1'])

            print(f"✅ Fold {fold} Results → AUC: {result['auc']}, ACC: {result['acc']}, F1: {result['f1']}")

        # Compute and save summary
        summary = {
            "best_hyperparameters_per_fold": all_best_params,
            "fold_auc_scores": all_fold_results,
            "summary_stats": {
                "auc_mean": np.mean(auc_list),
                "auc_std": np.std(auc_list),
                "acc_mean": np.mean(acc_list),
                "acc_std": np.std(acc_list),
                "f1_mean": np.mean(f1_list),
                "f1_std": np.std(f1_list)
            }
        }

        output_dir = os.path.join(RESULTS_ROOT, f"{source}_{target}")
        os.makedirs(output_dir, exist_ok=True)
        # Include the folds in the filename: parallel workers invoke this script
        # once per fold, and a fixed name would make each invocation overwrite
        # the previous fold's summary for the same pair.
        fold_tag = "_folds" + "-".join(str(f) for f in folds)
        file_path = os.path.join(
            output_dir, f"evaluation_summary_{source}_{target}{fold_tag}_results.json")
        with open(file_path, 'w') as fp:
            json.dump(summary, fp, indent=2)

        print(f"📁 Saved summary: {file_path}")

