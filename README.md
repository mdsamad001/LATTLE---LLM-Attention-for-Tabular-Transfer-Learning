# LATTLE: LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains

This is the official implementation.

---

## 📁 Project Structure

```
LATTLE/
│
├── LLM_finetune_module/
│   ├── Finetune-LLM.py         # Fine-tunes LLM on a source dataset
│   ├── models.py               # Custom model components
│   └── utils_pickle.py         # Pickle-related utilities
│
├── transfer_learning_module/
│   └── lattle/                 # LATTLE implementation (focus on modeling_lattle.py, modeling_lattle_GPT.py, lattle.py)
│
├── run_script-LLM-gFTT.py      # Step 1: Fine-tune LLM on source dataset
├── save_KV_weights.py          # Step 2: Extract & save KV attention weights
├── run_script-LATTLE.py        # Step 3: Transfer learning on target dataset
├── environment.yaml            # Environment specification
└── README.md                   # This file
```

---

## 🚀 How to Run the Pipeline

### Step 1: Fine-tune the LLM on a Source Dataset

Run the script below to fine-tune a GPT2-based model on your selected source tabular dataset.

```bash
python run_script-LLM-gFTT.py
```

This uses modules from `LLM_finetune_module` and saves the fine-tuned model for downstream use.

---

### Step 2: Save Key-Value Attention Weights

After fine-tuning, run the following script to extract and save the attention **Key (K)** and **Value (V)** weights from the trained model.

```bash
python save_KV_weights.py
```

This will generate and store the KV weights that will be transplanted into the target model.

---

### Step 3: Transfer Learning on Target Dataset using KV Transplant

Use the saved KV weights to perform attention-guided transfer learning on the target dataset.

```bash
python run_script-LATTLE.py
```

This script leverages the pre-extracted KV weights for transfer learning via the LATTLE strategy.

---

## 📦 Requirements

Install dependencies with:

```bash
conda env create -f environment.yaml
conda activate lattle
```

---

## 📊 Datasets

We use tabular datasets from OpenML (e.g., `credit-g`, `car`, `churn`, `sick`, etc.). You can customize the dataset list in `run_script-LLM-gFTT.py`.

Below are example source-target dataset pairs used for transfer experiments:

```
("credit-g", "Diabetes")
 ("Cardivascular-Disease", "Diabetes")
 ("optdigits", "vehicle")
 ("mfeat-fourier", "vehicle")
 ("sick", "cmc")
 ("churn", "cmc")
 ("car-evaluation", "pc1")
 ("steel-plates-fault", "pc1")
 ("steel-plates-fault", "cylinder-bands")
 ("seismic-bumps", "cylinder-bands")
```
---

## 🙏 Acknowledgements

This project builds upon the following open-source repositories:

- [TransTab](https://github.com/RyanWangZf/transtab): TransTab: A flexible transferable tabular learning framework.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): For pretrained LLM components.


