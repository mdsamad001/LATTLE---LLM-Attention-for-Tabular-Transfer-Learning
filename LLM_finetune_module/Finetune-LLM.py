# === Standard Library ===
import argparse
import json
import os
import random
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from transformers import (
    EarlyStoppingCallback,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import utils
from LLM_finetune_module.models import DistilGPT2WithFTTHead


parser = argparse.ArgumentParser()


parser.add_argument('--dataset', required=True, type=str,
                    choices=[
                        'breast-cancer',
                        'sepsis',
                        'yeast',
                        'dermatology',
                        'myocardial',
                        'bank',
                        'credit-g',
                        'adult',
                        'heart',
                        'communities',
                        'car',
                        'blood',
                        'diabetes',
                        'blood-transfusion-service-center',
                        'steel-plate-fault',
                        'pc3',
                        'kc1',
                        'GesturePhaseSegmentationProcessed',
                        'churn',
                        'Cardiovascular-Disease-dataset',
                        'optdigits',
                        'sick',
                        'seismic_bumps',
                        'mfeat-fourier',
    
                    ])
parser.add_argument('--batch', default=16, type=int)
parser.add_argument('--shot', default=4, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--prompt_template', default=1, type=int)
parser.add_argument('--name', default='Distil-GPT', type=str)
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')

args = parser.parse_args()

report_to_value = "wandb" if args.use_wandb else "none"

# Set the seed for reproducibility (use args.seed everywhere so model init
# and training randomness vary across seeds, not just the data split)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

shot = args.shot
seed = args.seed
dataset = args.dataset

# If using a GPU, ensure reproducibility by setting CUDA seeds as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


start_time = time()


from transformers import TrainerCallback

class TrainingMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):
        # Evaluate model on training dataset
        if train_dataloader is not None:
            model.eval()
            all_preds = []
            all_labels = []
            for batch in train_dataloader:
                inputs = {key: val.to(args.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(args.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
            
            # Compute accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Training Accuracy for Epoch {state.epoch}: {accuracy}")
            state.log_history.append({'epoch': state.epoch, 'train_accuracy': accuracy})



# Load dataset
utils.set_seed(seed)
df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(dataset, shot, seed, full_shot = True, cv = False)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=1/8, random_state=seed)

target_col = target_attr

if args.prompt_template == 1:
    # Define the serialization function
    def serialize_row(row):
        # print(row)
        row_str = '. '.join([f'{k}: {v}' for k,v in row.items()])
        target = f'What is the {target_col}?'
        return f'{row_str}. {target}'
    
    
    train_df = X_train
    val_df = X_val
    test_df = X_test
    
    # Apply serialization
    train_df['serialized_text'] = train_df.apply(serialize_row, axis=1)
    val_df['serialized_text'] = val_df.apply(serialize_row, axis=1)
    test_df['serialized_text'] = test_df.apply(serialize_row, axis=1)
    
    # Convert to Hugging Face datasets
    from datasets import Dataset
    
    train_dataset = Dataset.from_pandas(train_df[['serialized_text']])
    
    val_dataset = Dataset.from_pandas(val_df[['serialized_text']])
    
    test_dataset = Dataset.from_pandas(test_df[['serialized_text']])
    
elif args.prompt_template == 2:
    def load_feature_descriptions(dataset_name):
        metadata_file = f'./data/{dataset_name}-metadata.json'
        with open(metadata_file, 'r') as f:
            feature_descriptions = json.load(f)
            print(feature_descriptions)
        return feature_descriptions
    
    # Define the new serialization function
    def serialize_row_new_format(row, feature_descriptions):
        task = f'Task: predict the class?'
        feature_desc = "Feature description:\n" + '\n'.join([f"{k}: {v}" for k, v in feature_descriptions.items()])
        sample = "Sample:\n" + '. '.join([f'{k} is {v}' for k, v in row.items()])
        return f'{task}\n\n{feature_desc}\n\n{sample}'

    # Load the feature descriptions based on the dataset name
    feature_descriptions = load_feature_descriptions(dataset)

    train_df = X_train
    val_df = X_val
    test_df = X_test
    
    # Apply new serialization format
    train_df['serialized_text'] = train_df.apply(lambda row: serialize_row_new_format(row, feature_descriptions), axis=1)
    val_df['serialized_text'] = val_df.apply(lambda row: serialize_row_new_format(row, feature_descriptions), axis=1)
    test_df['serialized_text'] = test_df.apply(lambda row: serialize_row_new_format(row, feature_descriptions), axis=1)
    # print(len(train_df['serialized_text']), val_df['serialized_text'])
    # Convert to Hugging Face datasets
    from datasets import Dataset
    
    train_dataset = Dataset.from_pandas(train_df[['serialized_text']])
    val_dataset = Dataset.from_pandas(val_df[['serialized_text']])
    test_dataset = Dataset.from_pandas(test_df[['serialized_text']])
    

# Step 1: Initialize tokenizer
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load pre-trained GPT2 classifier
base_model = GPT2ForSequenceClassification.from_pretrained(
    model_name, num_labels=len(np.unique(y_train))
)

for name, param in base_model.transformer.named_parameters():
    # Only unfreeze the parameters of the last transformer block (usually named 'h.5' in DistilGPT-2)
    if 'h.5' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        
# Step 3: Set pad token id
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.resize_token_embeddings(len(tokenizer))

# Instantiate the model
model = DistilGPT2WithFTTHead(
    base_model=base_model,
    num_class=len(np.unique(y_train)),  # or 2 for binary classification
    hidden_dim=768,
    ftt_dim=128,
    num_heads=8,
    dropout=0.1
)


import copy
from collections import defaultdict

# Store the cumulative L2 norm of updates for each layer
cumulative_updates = defaultdict(float)
previous_weights = {}

# Initialize with the starting weights
for i, block in enumerate(model.base.h):
    for name, param in block.named_parameters():
        previous_weights[f"h.{i}.{name}"] = param.detach().clone()

train_token_lengths = train_df['serialized_text'].apply(lambda x: len(tokenizer.tokenize(x)))
val_token_lengths = val_df['serialized_text'].apply(lambda x: len(tokenizer.tokenize(x)))

# Find the maximum token length using train/val only (test set must not
# influence preprocessing)
max_train_tokens = train_token_lengths.max()
max_val_tokens = val_token_lengths.max()

# Determine the overall maximum token length
max_tokens = max(max_train_tokens, max_val_tokens)



max_length_allowed = base_model.config.n_positions  # Usually 1024 for GPT-2

# If your max token length exceeds this, you need to truncate the sequences
if max_tokens > max_length_allowed:
    print(f"Warning: Max token length exceeds {max_length_allowed}, truncating sequences")
    max_tokens = max_length_allowed


max_tokens = int(max_tokens)
train_encodings = tokenizer(
    train_df['serialized_text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=max_tokens
)

val_encodings = tokenizer(
    val_df['serialized_text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=max_tokens
)

test_encodings = tokenizer(
    test_df['serialized_text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=max_tokens
)


# Convert encoded labels to tensors
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

train_labels = torch.tensor(y_train_encoded)
val_labels = torch.tensor(y_val_encoded)
test_labels = torch.tensor(y_test_encoded)


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}  # Directly access val[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TabularDataset(train_encodings, train_labels)
val_dataset = TabularDataset(val_encodings, val_labels)
test_dataset = TabularDataset(test_encodings, test_labels)

import inspect
import math

import transformers

NUM_EPOCHS = 200   # paper Sec 4.1: source LLM fine-tuned for 200 epochs
# transformers >=5.15 removed `warmup_ratio`; warmup_steps works on all versions
_steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch))
_warmup_steps = int(0.1 * _steps_per_epoch * NUM_EPOCHS)

_ta_kwargs = dict(
    output_dir=f'./llm_finetuning_results/{args.name}_{args.prompt_template}_{args.dataset}_{args.seed}',
    run_name=model_name + '-' + args.name + '-' + dataset + '-' + str(seed),
    eval_strategy="epoch",               # evaluate at the end of each epoch
    logging_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,         # upper bound; early stopping picks the best epoch
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    warmup_steps=_warmup_steps,          # == warmup_ratio 0.1, computed for portability
    weight_decay=0.01,
    logging_dir='./llm_finetuning_results/logs',   # dropped in transformers >=5.15
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="eval_auc",    # select on AUC, the reported metric
    greater_is_better=True,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    report_to=report_to_value,
)

# TrainingArguments drops/renames arguments between transformers releases.
# Keep only what this installed version accepts, and say what was dropped.
_ta_supported = set(inspect.signature(TrainingArguments.__init__).parameters)
_dropped = sorted(set(_ta_kwargs) - _ta_supported)
if _dropped:
    print(f"[compat] transformers {transformers.__version__} does not accept "
          f"{_dropped}; continuing without them.")
    _ta_kwargs = {k: v for k, v in _ta_kwargs.items() if k in _ta_supported}

training_args = TrainingArguments(**_ta_kwargs)

if args.use_wandb:
    if wandb is None:
        raise ImportError("wandb is not installed but --use_wandb was passed")
    # Authenticate via `wandb login` or the WANDB_API_KEY env var — never
    # hardcode API keys in source.
    wandb.login()
    wandb.init(
        project="distilgpt2-tabular-classification-gftt-user",
        name=model_name + '-' + args.name + '-' + dataset + '-' + str(seed)
    )

def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Compute accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # Handle AUC for multi-class and binary-class cases
    try:
        # Convert predictions to probabilities using softmax
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)

        if probs.shape[1] == 2:  # Binary classification case
            auc = roc_auc_score(labels, probs[:, 1])  # AUC for positive class
        else:
            # Ensure labels and probabilities have the same number of classes
            num_classes = probs.shape[1]
            unique_labels = len(np.unique(labels))
            
            if num_classes != unique_labels:
                print(f"Warning: {unique_labels} classes in labels, but {num_classes} columns in predictions.")
                # Adjust the number of columns in predictions if necessary by padding missing classes
                new_probs = np.zeros((len(probs), unique_labels))
                new_probs[:, :num_classes] = probs.numpy()
                auc = roc_auc_score(labels, new_probs, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except ValueError as e:
        # 0.0 rather than NaN: eval_auc drives checkpoint selection, and NaN
        # comparisons are always False, which would freeze "best model" at the
        # first epoch.
        print(f"AUC computation failed: {e}")
        auc = 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc  # Add AUC to the returned metrics
    }

# Initialize Trainer with attention mask handling
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # evaluate on validation test. Using test set means we leaked information.
    eval_dataset=val_dataset,  
    compute_metrics=compute_metrics,
    callbacks=[TrainingMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=10)],
)

# Train the model
train_output = trainer.train()

save_dir = f"./llm_finetuning_results/best_models/{args.name}_{args.prompt_template}_{args.dataset}_{args.seed}"
os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure directory exists

# Save the fine-tuned GPT2 transformer via save_pretrained (writes config.json
# + weights with proper key names) so it can be reloaded with
# GPT2Model.from_pretrained in save_KV_weights.py. The wrapper's full state
# dict is saved separately (NOT as pytorch_model.bin, which would shadow the
# HF weights at load time).
trainer.model.base.save_pretrained(os.path.join(save_dir, "base"))
torch.save(trainer.model.state_dict(), os.path.join(save_dir, "wrapper_state_dict.pt"))


# os.makedirs('./llm_finetuning_results/plots', exist_ok=True)

# Log the metrics
metrics = train_output.metrics
print(f"Training metrics: {metrics}")

# Get the logs for accuracy and loss for both training and validation
training_logs = trainer.state.log_history

# Check the log history to see if the losses and accuracy are logged correctly
# print("Trainer log history: ", trainer.state.log_history)

# Extract accuracy and loss from logs
train_accuracy = [log.get('train_accuracy') for log in training_logs if 'train_accuracy' in log]
val_accuracy = [log.get('eval_accuracy') for log in training_logs if 'eval_accuracy' in log]
train_loss = [log.get('loss') for log in training_logs if 'loss' in log]
val_loss = [log.get('eval_loss') for log in training_logs if 'eval_loss' in log]

# Debugging: Check lengths of lists
print(f"Training Accuracy: {len(train_accuracy)}, Validation Accuracy: {len(val_accuracy)}")
print(f"Training Loss: {len(train_loss)}, Validation Loss: {len(val_loss)}")


# log_data = {}
# def save_training_plots_and_logs(train_loss, val_loss, train_acc, val_acc, epochs, args, save_dir='./plot_figure/plots'):
#     os.makedirs(save_dir, exist_ok=True)
    
    # Define filenames
    # loss_plot_name = f'loss_curve_{args.dataset}_{args.name}_{args.prompt_template}_seed_{args.seed}.pdf'
    # acc_plot_name = f'accuracy_curve_{args.dataset}_{args.name}_{args.prompt_template}_seed_{args.seed}.pdf'
    # csv_name = f'metrics_log_{args.dataset}_{args.name}_{args.prompt_template}_seed_{args.seed}.csv'
    # json_name = f'training_metrics_{args.dataset}_{args.name}_{args.prompt_template}_{args.seed}.json'

    # # Plot loss
    # plt.figure(figsize=(10, 6))
    # min_size = min(len(epochs), len(train_loss))
    # plt.plot(epochs[:min_size], train_loss[:min_size], label='Training Loss')

    # if val_loss:
    #     min_size = min(len(epochs), len(val_loss))
    #     plt.plot(epochs[:min_size], val_loss[:min_size], label='Validation loss')

    # plt.xlabel('Epochs', fontsize=18)
    # plt.ylabel('Loss', fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=16)
    # plt.grid(False)
    # plt.tight_layout()
    # # plt.savefig(os.path.join(save_dir, loss_plot_name))
    # plt.close()

    # Plot accuracy
    # plt.figure(figsize=(10, 6))
    # min_size = min(len(epochs), len(train_acc))
    # plt.plot(epochs[:min_size], train_acc[:min_size], label='Training Accuracy')

    # if val_acc:
    #     min_size = min(len(epochs), len(val_acc))
    #     plt.plot(epochs[:min_size], val_acc[:min_size], label='Validation Accuracy')

    # plt.xlabel('Epochs', fontsize=18)
    # plt.ylabel('Accuracy', fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=16)
    # plt.grid(False)
    # plt.tight_layout()
    # # plt.savefig(os.path.join(save_dir, acc_plot_name))
    # plt.close()

    # Save metrics as CSV
    # max_len = max(len(epochs), len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    # data = {
    #     'epoch': epochs[:max_len],
    #     'train_loss': train_loss[:max_len] + [None]*(max_len - len(train_loss)),
    #     'val_loss': val_loss[:max_len] + [None]*(max_len - len(val_loss)),
    #     'train_accuracy': train_acc[:max_len] + [None]*(max_len - len(train_acc)),
    #     'val_accuracy': val_acc[:max_len] + [None]*(max_len - len(val_acc)),
    # }
    # df = pd.DataFrame(data)
    # df.to_csv(os.path.join(save_dir, csv_name), index=False)

    # Save raw logs as JSON
    # log_data = {
    #     "train_loss": train_loss,
    #     "val_loss": val_loss,
    #     "train_accuracy": train_acc,
    #     "val_accuracy": val_acc,
    # }
    # with open(os.path.join(save_dir, json_name), 'w') as f:
    #     json.dump(log_data, f, indent=4)



# if len(train_accuracy) > 0 and len(train_loss) > 0:
#     epochs = list(range(1, len(train_loss) + 1))
#     save_training_plots_and_logs(train_loss, val_loss, train_accuracy, val_accuracy, epochs, args)
# else:
#     print("Error: Training or validation accuracy/loss lists are empty. Check log history for details.")



from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Get predictions
predictions = trainer.predict(test_dataset)

# Get probabilities for the positive class
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)

multiclass = probs.shape[1] > 2

if multiclass == False:
    result_auc = roc_auc_score(y_test, probs[:, 1])
else:
    result_auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')


result_f1 = f1_score(y_test_encoded, probs.argmax(dim=1), average='weighted')
result_accuracy = accuracy_score(y_test_encoded, probs.argmax(dim=1))
time = time() - start_time


log_data = {
    "metrics": metrics,  # Training metrics like total runtime, loss, etc.
    "training_logs": training_logs,  # Log history including loss, accuracy, etc.
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy,
    "train_loss": train_loss,
    "val_loss": val_loss,
    'test_auc_score': result_auc,
    'test_f1_score': result_f1,
    'test_accuracy': result_accuracy,
}

# Save to JSON file
# output_file = f'./llm_finetuning_results/plots/training_metrics_{args.dataset}_{args.name}_{args.prompt_template}_{args.seed}.json'
# with open(output_file, 'w') as f:
#     json.dump(log_data, f, indent=4)
# predictions.predictions
print(f'AUC: {result_auc}')
print(f'F1: {result_f1}')
print(f'ACC: {result_accuracy}')
print(f'Time: {time}')



