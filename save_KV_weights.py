import torch
import os
import glob
from transformers import GPT2Model


def extract_final_gpt2_kv_weights(gpt2_model):
    """
    Extracts Key and Value projection weights from the final transformer block
    of a (fine-tuned) GPT2 model.

    GPT2's fused c_attn is a Conv1D with weight shape [hidden_dim, 3*hidden_dim]
    (computed as x @ W). Transposing gives the standard nn.Linear layout
    [3*hidden_dim, hidden_dim]; rows [d:2d] are K and rows [2d:] are V.
    """
    final_block = gpt2_model.h[-1]
    attention = final_block.attn

    c_attn_weight = attention.c_attn.weight.detach().cpu().T  # [3*hidden, hidden]

    hidden_dim = c_attn_weight.shape[0] // 3
    W_K = c_attn_weight[hidden_dim:2*hidden_dim, :].clone()
    W_V = c_attn_weight[2*hidden_dim:, :].clone()
    return W_K, W_V


# ---------------- Main script ---------------- #
if __name__ == "__main__":
    dataset_list = [
        'credit-g',
        'car',
        'sick',
        'optdigits',
        'mfeat-fourier',
        'churn',
        'steel-plate-fault',
        'seismic_bumps',
        'Cardiovascular-Disease-dataset',
    ]

    seeds = list(range(3))
    base_save_path = "./transfer_learning_module/gpt_FTTs_GPT-last_layer_KV_weights"

    for dataset_name in dataset_list:
        for seed in seeds:
            print(f"\n Processing: {dataset_name} | Seed: {seed}")
            checkpoint_pattern = f'./llm_finetuning_results/best_models/distilgpt-gftt-last_1_{dataset_name}_{seed}'
            checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
            if not checkpoint_paths:
                print(f"  ❌ No checkpoint found: {checkpoint_pattern}")
                continue

            checkpoint_path = checkpoint_paths[-1]
            base_path = os.path.join(checkpoint_path, "base")
            if not os.path.exists(os.path.join(base_path, "config.json")):
                print(f"  ❌ No saved base model at {base_path} — re-run fine-tuning "
                      f"(Finetune-LLM.py now saves the GPT2 transformer via save_pretrained).")
                continue
            print(f"  Found checkpoint: {base_path}")

            gpt2_model = GPT2Model.from_pretrained(base_path)

            # Extract weights from final GPT2 layer
            K_final, V_final = extract_final_gpt2_kv_weights(gpt2_model)

            # Save
            save_dir = os.path.join(base_save_path, f"{dataset_name}_{seed}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(K_final, os.path.join(save_dir, "K_final.pt"))
            torch.save(V_final, os.path.join(save_dir, "V_final.pt"))

            print(f"  Saved K, V to → {save_dir}")
