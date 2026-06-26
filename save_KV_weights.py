import torch
import os
import glob
from transformers import GPT2Tokenizer
from LLM_finetune_module.models import DistilGPT2WithFTTHead
from transformers import GPT2ForSequenceClassification


def load_custom_model(checkpoint_path, tokenizer, num_labels=2):
    # This loads the pretrained weights automatically
    base_model = GPT2ForSequenceClassification.from_pretrained(checkpoint_path)

    # Wrap with your custom FTT head
    custom_model = DistilGPT2WithFTTHead(
        base_model=base_model,
        num_class=num_labels,
        hidden_dim=768,
        ftt_dim=128,
        num_heads=8,
        dropout=0.1
    )
    return custom_model

def extract_final_distilgpt2_kv_weights(model):
    """
    Extracts Key and Value projection weights from the final transformer block of the DistilGPT2 base model.
    """
    final_block = model.base.h[-1]  # DistilGPT2 transformer blocks are in model.base.h
    attention = final_block.attn

    # Transpose so shape becomes [3 * hidden_dim, hidden_dim]
    c_attn_weight = attention.c_attn.weight.detach().cpu().T  # Transpose!

    hidden_dim = c_attn_weight.shape[0] // 3
    W_K = c_attn_weight[hidden_dim:2*hidden_dim, :]
    W_V = c_attn_weight[2*hidden_dim:, :]
    return W_K, W_V



def extract_cls_embedding(model, tokenizer):
    model.eval()
    dummy_input = tokenizer("This is a placeholder input.", return_tensors="pt", padding=True, truncation=True)
    dummy_input = {k: v.to(next(model.parameters()).device) for k, v in dummy_input.items()}

    with torch.no_grad():
        outputs = model.base(**dummy_input)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # (hidden_dim,)
    return cls_embedding


# ---------------- Main script ---------------- #
if __name__ == "__main__":
    dataset_list = [
        # 'sick',
        # 'seismic_bumps',
        # 'mfeat-fourier',
        'credit-g',
        # 'car',
        # 'steel-plate-fault',
        # 'churn',
        # 'Cardiovascular-Disease-dataset',
        # 'optdigits',
    ]
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    seeds = list(range(10))
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
            print(f"  Found checkpoint: {checkpoint_path}")

            model = load_custom_model(checkpoint_path, tokenizer)

            # Extract weights from final DistilGPT2 layer
            K_final, V_final = extract_final_distilgpt2_kv_weights(model)
            CLS_embedding = extract_cls_embedding(model, tokenizer)

            # Save
            save_dir = os.path.join(base_save_path, f"{dataset_name}_{seed}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(K_final, os.path.join(save_dir, f"K_final.pt"))
            torch.save(V_final, os.path.join(save_dir, f"V_final.pt"))
            torch.save(CLS_embedding, os.path.join(save_dir, f"CLS_final.pt"))

            print(f"  Saved K, V, CLS to → {save_dir}")
