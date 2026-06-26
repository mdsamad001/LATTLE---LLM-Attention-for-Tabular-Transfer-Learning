import pandas as pd
import os
from itertools import product
from LLM_finetune_module.utils_pickle import *


dataset_list = [
        'credit-g',
    #     'car',
    #     'steel-plate-fault',
    #     'churn',
    #     'optdigits',
    #     'sick',
    #     'seismic_bumps',
    #     'mfeat-fourier',
    # 'Cardiovascular-Disease-dataset'
]

seed_list = list(range(0,1))  
model_files = {
    'Distil-GPT2-simple': '-m LLM_finetune_module.Finetune-LLM',
}

def get_results(model, dataset, seed, bs=16, pt=1, name=None, use_wandb= False):
    run_args = [
        'python', model_files[model],
        '--dataset', dataset,
        '--seed', seed,
        '--batch', bs,
        '--prompt_template', pt,
        '--name', name,
    ]    
    if use_wandb:
        run_args.append('--use_wandb')
        
    print("💡 WandB is disabled by default.")
    print("➡️  If you want to enable Weights & Biases logging, set `use_wandb = True` in the 'run_script-LLM-gFTT.py'  script.\n")

    run_args = [str(i) for i in run_args]
    command = ' '.join(run_args)
    print(command)

    file_name = f"./cached_results/{'_'.join(run_args)}.pkl"

    loaded = False

    if not loaded:
        r = os.popen(command).readlines()
        save_var(r, file_name)

    # print(r)

    # Initialize
    auc_ensemble = f1_ensemble = runtime = None

    # Parse result lines by key
    for line in r:
        if line.strip().startswith("AUC:"):
            try:
                auc_ensemble = float(line.strip().split(":")[1])
            except:
                pass
        elif line.strip().startswith("F1:"):
            try:
                f1_ensemble = float(line.strip().split(":")[1])
            except:
                pass
        elif line.strip().startswith("Time:"):
            try:
                runtime = float(line.strip().split(":")[1])
            except:
                pass

    if auc_ensemble is None or f1_ensemble is None or runtime is None:
        print("Failed to extract AUC, F1, or Time. Full output:")
        print("".join(r))
        return [], None, None, None

    return [], auc_ensemble, f1_ensemble, runtime


if __name__ != '__main__':
    exit()

prompt_template = [1]
name = ['distilgpt-gftt-last']
combos = list(product(model_files.keys(), dataset_list, name, prompt_template, seed_list))

cols = ['model', 'dataset', 'shot', 'seed', 'ensemble', 'auc_list', 'auc_ensemble', 'f1', 'time']
rows = []

bs = 16
prompt_template = prompt_template[0]
model = 'distil-gpt'
use_wandb=False
# csv_file_name = f'./llm_finetuning_results/expt-{model}-paper-finetune-{prompt_template}_patience-10-bs-{bs}_ftt_nonlinear-last_layer.csv' ## benchmark for paper on featllm
pt_d = {1:'', 2:'metadata'}
for model, dataset, name, pt, seed in combos:
    _, _, _, _ = get_results(model, dataset, seed, bs = bs, pt = pt, name = name, use_wandb = use_wandb)
    # print(auc_fold, auc_ensemble, f1, time)
    # rows.append([model.split('-')[0]+'_'+name+'_'+pt_d[pt], dataset, 'full', seed, 1, auc_fold, auc_ensemble, f1, time])
    
    
# df = pd.DataFrame(rows, columns=cols)
# df.to_csv(csv_file_name, index = 0)