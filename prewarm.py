#!/usr/bin/env python3
"""Pre-populate shared caches before spawning parallel workers.

Concurrent first-touch of the HF hub cache, the BERT tokenizer directory, or
the OpenML dataset pickles would race between workers, so warm them once.
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, BertTokenizerFast

print("warming distilgpt2 ...")
GPT2Tokenizer.from_pretrained("distilgpt2")
GPT2ForSequenceClassification.from_pretrained("distilgpt2", num_labels=2)

print("warming bert-base-uncased tokenizer ...")
if not os.path.exists('./lattle_tokenizer/tokenizer'):
    tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tok.save_pretrained('./lattle_tokenizer/tokenizer')

from transfer_learning_module import lattle

TARGETS = ["ilpd", "vehicle", "cmc", "pc1", "cylinder-bands"]
FOLDS = int(os.environ.get("NFOLDS", "10"))
os.makedirs("cached_datasets", exist_ok=True)

for name in TARGETS:
    for seed in range(FOLDS):
        path = f"cached_datasets/{name}_seed{seed}.pkl"
        if os.path.exists(path):
            continue
        print(f"  caching {name} seed {seed} ...", flush=True)
        res = lattle.load_data([name], seed=seed)
        with open(path, "wb") as f:
            pickle.dump(res, f)

print("PREWARM_DONE")
