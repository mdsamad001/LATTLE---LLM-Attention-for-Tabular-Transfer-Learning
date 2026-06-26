import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import openml

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes if num_classes > 2 else 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid() if num_classes == 2 else nn.Identity()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)

def load_pretrained_mlp(filepath, input_size, num_classes, freeze_layers=True, device='cuda'):
    """Load a pre-trained MLP model and adjust input size dynamically during fine-tuning."""

    # Load original state_dict
    state_dict = torch.load(filepath, map_location=device)

    # Extract the source input size from saved weights (fc1 weight shape)
    source_input_size = state_dict['fc1.weight'].shape[1]  # [128, input_size]

    # Reinitialize the model with the new target input size
    model = MLP(input_size=input_size, num_classes=num_classes)

    # Update all layers except `fc1`
    updated_state_dict = state_dict.copy()
    del updated_state_dict['fc1.weight']
    del updated_state_dict['fc1.bias']

    # Load remaining weights
    model.load_state_dict(updated_state_dict, strict=False)  # Ignore fc1 mismatch

    model.to(device)
    return model


def save_checkpoint(model, optimizer, input_size, num_classes, epoch, filepath='mlp_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")
    
# MLP Training Function
def train_mlp(model, criterion, optimizer, train_loader, val_loader, input_size=None, num_classes=None, num_epochs=50, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            #print(outputs)
            loss = criterion(outputs, batch_y)
            #print(loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    save_checkpoint(model, optimizer, input_size, num_classes, num_epochs)
    
    return model

# MLP Evaluation Function
def evaluate_mlp(model, test_loader, num_classes, device='cpu'):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            if num_classes == 2:
                preds = torch.sigmoid(outputs).cpu().numpy()
            else:
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    if num_classes == 2:
        auc_score = roc_auc_score(all_labels, np.array(all_preds))
    else:
        auc_score = roc_auc_score(all_labels, np.array(all_preds), multi_class="ovr", average="macro")

    return auc_score

# Full Pipeline for MLP
def load_data_mlp(dataname, seed=42, dataset_config=None, batch_size=128):
    print('Running MLP')

    # Load the dataset
    dataset = openml.datasets.get_dataset(dataname)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute)
    
    num_classes = y.nunique()
    
    # Convert y labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y) 

    # Identify categorical and numerical columns
    all_cols = np.array(attribute_names)
    categorical_indicator = np.array(categorical_indicator)
    cat_cols = list(all_cols[categorical_indicator])
    num_cols = list(all_cols[~categorical_indicator])

    if dataset_config is not None:
        bin_cols = [c for c in cat_cols if c in dataset_config.get('bin', [])]
    else:
        bin_cols = []

    cat_cols = [c for c in cat_cols if c not in bin_cols]

    # Random Train/Test Split using the seed
    train_dataset, test_dataset, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Further split training data into train and validation
    train_dataset, val_dataset, y_train, y_val = train_test_split(
        train_dataset, y_train, test_size=0.125, stratify=y_train, random_state=seed
    )  # 12.5% of train -> validation (final split: 70% train, 10% val, 20% test)

    # Standardize numerical features
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        num_imputer.fit(train_dataset[num_cols])

        train_dataset[num_cols] = num_imputer.transform(train_dataset[num_cols])
        test_dataset[num_cols] = num_imputer.transform(test_dataset[num_cols])
        val_dataset[num_cols] = num_imputer.transform(val_dataset[num_cols])
        
        scaler = StandardScaler()
        scaler.fit(train_dataset[num_cols])

        train_dataset.loc[:, num_cols] = scaler.transform(train_dataset[num_cols])
        test_dataset.loc[:, num_cols] = scaler.transform(test_dataset[num_cols])
        val_dataset.loc[:, num_cols] = scaler.transform(val_dataset[num_cols])

    # One-Hot Encode categorical features
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_imputer.fit(train_dataset[cat_cols])
        
        train_dataset[cat_cols] = cat_imputer.transform(train_dataset[cat_cols])
        test_dataset[cat_cols] = cat_imputer.transform(test_dataset[cat_cols])
        val_dataset[cat_cols] = cat_imputer.transform(val_dataset[cat_cols])
        
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
        encoder.fit(train_dataset[cat_cols])

        # Transform categorical features
        train_cat = encoder.transform(train_dataset[cat_cols])
        val_cat = encoder.transform(val_dataset[cat_cols])
        test_cat = encoder.transform(test_dataset[cat_cols])

        # Convert to DataFrame
        train_cat_df = pd.DataFrame(train_cat, index=train_dataset.index, columns=encoder.get_feature_names_out(cat_cols))
        val_cat_df = pd.DataFrame(val_cat, index=val_dataset.index, columns=encoder.get_feature_names_out(cat_cols))
        test_cat_df = pd.DataFrame(test_cat, index=test_dataset.index, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns
        train_dataset = train_dataset.drop(columns=cat_cols)
        val_dataset = val_dataset.drop(columns=cat_cols)
        test_dataset = test_dataset.drop(columns=cat_cols)

        # Concatenate the one-hot encoded features
        train_dataset = pd.concat([train_dataset, train_cat_df], axis=1)
        val_dataset = pd.concat([val_dataset, val_cat_df], axis=1)
        test_dataset = pd.concat([test_dataset, test_cat_df], axis=1)

    # Convert to PyTorch Tensors
    train_tensor = torch.tensor(train_dataset.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_dataset.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_dataset.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if num_classes == 2 else torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if num_classes == 2 else torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if num_classes == 2 else torch.long)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = MLP(input_size=train_dataset.shape[1], num_classes=num_classes)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    input_size=train_dataset.shape[1]
    
    # Train and Evaluate Model
    #model = train_mlp(model, criterion, optimizer, train_loader, val_loader, input_size, num_classes, num_epochs, device)
    #auc_score = evaluate_mlp(model, test_loader, num_classes, device)

    #print(f"AUC score: {auc_score}")
    return model, train_loader, val_loader, test_loader, input_size, num_classes, criterion, optimizer


