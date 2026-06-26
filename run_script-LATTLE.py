#!/usr/bin/env python3
"""
Script to run hyperparameter tuning and evaluation across multiple dataset pairs
using TransTab and Optuna, then save AUC, ACC, and F1 summaries.
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



# === Configuration ===
dataset_pairs = [
    ("credit-g", "Diabetes"),
    # ("Cardivascular-Disease", "Diabetes"),
    # ("optdigits", "vehicle"),
    # ("mfeat-fourier", "vehicle"),
    # ("sick", "cmc"),
    # ("churn", "cmc"),
    # ("car-evaluation", "pc1"),
    # ("steel-plates-fault", "pc1"),
    # ("steel-plates-fault", "cylinder-bands"),
    # ("seismic-bumps", "cylinder-bands")
]

num_class_target_map = {
    "Diabetes": 2,
    "vehicle": 4,
    "cmc": 3,
    "pc1": 2,
    "cylinder-bands": 2
}

folds = list(range(1))
KV_DIR = "./transfer_learning_module/gpt_FTTs_GPT-last_layer_KV_weights"
RESULTS_ROOT = "./transfer_learning_results/optuna_ckpt_results"




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

    model = lattle.build_classifier(
        cat_cols, num_cols, bin_cols,
        num_class=num_class_target, p=1
    )

    # Apply pre-trained CA weights if available
    k_path = os.path.join(KV_DIR, f"{source}_{fold}/K_final.pt")
    v_path = os.path.join(KV_DIR, f"{source}_{fold}/V_final.pt")
    if os.path.exists(k_path) and os.path.exists(v_path):
        model.apply_ca(torch.load(k_path), torch.load(v_path))
    else:
        print('could not find the pretrain-weight')
        exit()

    # Train on validation set for hyperparameter evaluation
    lattle.train(
        model,
        (X_train, y_train), (X_val, y_val),
        num_epoch=1,
        eval_metric='auc',
        eval_less_is_better=False,
        output_dir='./transfer_learning_results/trials',
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dropout_prob=hidden_dropout_prob,
        num_class=num_class_target,
        save_best=False,
        exp=f'{source}_{target}_fold_{fold}_optuna',
        warmup_ratio=warmup_ratio,
    )

    # Predict and return AUC
    y_pred = lattle.predict(model, X_val, y_val)
    if num_class_target == 2:
        y_score = y_pred[:, 1] if y_pred.ndim == 2 else y_pred
        return roc_auc_score(y_val, y_score)
    else:
        return roc_auc_score(y_val, y_pred, multi_class="ovr", average="macro")


def evaluate_final(source, target, num_class_target, best_params, fold):
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

    # 5) rebuild your model and (optionally) apply CA weights
    model = lattle.build_classifier(cat_cols, num_cols, bin_cols,
                                      num_class=num_class_target, p=1)
    k_path = os.path.join(KV_DIR, f"{source}_{fold}/K_final.pt")
    v_path = os.path.join(KV_DIR, f"{source}_{fold}/V_final.pt")
    if os.path.exists(k_path) and os.path.exists(v_path):
        model.apply_ca(torch.load(k_path), torch.load(v_path))
    else:
        print('could not find the pretrain-weight')
        exit()
    # 6) train on the combined set, validate on test set
    lattle.train(
        model,
        (X_train_val, y_train_val),
        (X_te_df,   y_te_s),
        num_epoch=1,
        eval_metric='auc',
        eval_less_is_better=False,
        output_dir='./transfer_learning_results',
        seed=fold,
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        hidden_dropout_prob=best_params['hidden_dropout_prob'],
        num_class=num_class_target,
        save_best=True,
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
    for source, target in dataset_pairs:
        print(f"\n🔁 Running for: {source} → {target}")
        num_class_target = num_class_target_map[target]

        all_fold_results = {}
        all_best_params = {}
        auc_list, acc_list, f1_list = [], [], []

        for fold in folds:
            print(f"\n🔧 Tuning Fold {fold} for: {source} → {target}")
            # study = optuna.create_study(direction='maximize')

            
            # Optional: create a unique study name per fold
            study_name = f"{source}_{target}_fold_{fold}"
            
            # Create full directory
            optuna_dir = "./transfer_learning_results/optuna_studies"
            os.makedirs(optuna_dir, exist_ok=True)
            
            # Save DB in that location
            storage_path = f"sqlite:///{optuna_dir}/{study_name}.db"

            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                storage=storage_path,
                load_if_exists=True  # allows continuing from past runs
            )

            study.optimize(
                lambda tr: objective(tr, source, target, num_class_target, fold),
                n_trials=1
            )

            best_trial = study.best_trial
            best_params = best_trial.params
            all_best_params[f'fold_{fold}'] = best_params

            # result = evaluate_fold(source, target, num_class_target, best_params, fold)
            result = evaluate_final(source, target, num_class_target, best_params, fold)
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
        file_path = os.path.join(output_dir, f"evaluation_summary_{source}_{target}_results.json")
        with open(file_path, 'w') as fp:
            json.dump(summary, fp, indent=2)

        print(f"📁 Saved summary: {file_path}")

