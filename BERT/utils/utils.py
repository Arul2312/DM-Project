import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, precision_recall_curve,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

class DTIDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Target Sequence'] + " [SEP] " + row['SMILES']
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['Label'], dtype=torch.long)
        }

def get_device():
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def compute_class_weights(csv_file, label_col='Label', device=None):
    labels = pd.read_csv(csv_file)[label_col]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weights = torch.tensor(class_weights, dtype=torch.float)
    return weights.to(device) if device else weights

def optimal_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = optimal_f1_threshold(y_true, y_prob)
    y_pred = (np.array(y_prob) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pc': average_precision_score(y_true, y_prob),
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'optimal_threshold': threshold
    }
