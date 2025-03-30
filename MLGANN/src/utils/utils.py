import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, accuracy_score
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

import torch
import torch.nn.functional as F

# import torch.nn.functional as F

def dti_loss(zD_pos, zT_pos, zD_neg, zT_neg, margin=0.4, lambda1=0.7, lambda2=0.3):
    # Normalize embeddings
    zD_pos = F.normalize(zD_pos, p=2, dim=1)
    zT_pos = F.normalize(zT_pos, p=2, dim=1)
    zD_neg = F.normalize(zD_neg, p=2, dim=1)
    zT_neg = F.normalize(zT_neg, p=2, dim=1)
    
    # Calculate inner product similarities
    pos_sim = torch.sum(zD_pos * zT_pos, dim=1)
    neg_sim = torch.sum(zD_neg * zT_neg, dim=1)
    
    # Triplet margin loss with distance-based similarity
    triplet_loss = torch.mean(torch.relu(
        margin - pos_sim + neg_sim
    ))
    
    # Binary cross-entropy loss
    scores = torch.cat([pos_sim, neg_sim])
    labels = torch.cat([
        torch.ones_like(pos_sim),
        torch.zeros_like(neg_sim)
    ])
    bce_loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    # L2 regularization
    l2_reg = 0.01 * (
        torch.norm(zD_pos) + 
        torch.norm(zT_pos) + 
        torch.norm(zD_neg) + 
        torch.norm(zT_neg)
    ) / 4.0
    
    # Combined loss
    total_loss = lambda1 * bce_loss + lambda2 * triplet_loss + l2_reg
    
    return total_loss

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=1)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=1)

def f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def aupr(y_true, y_scores):
    return average_precision_score(y_true, y_scores)

def create_pyg_graph(adjacency_matrix, feature_matrix):
    edge_index = torch.nonzero(torch.tensor(adjacency_matrix)).t().contiguous()
    x = torch.tensor(feature_matrix, dtype=torch.float)

    # Ensure edges are within bounds
    max_index = x.shape[0] - 1
    edge_index = edge_index[:, (edge_index[0] <= max_index) & (edge_index[1] <= max_index)]

    return Data(x=x, edge_index=edge_index)

class DTIDataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        # Validate input pairs
        if not all(len(pair) >= 2 for pair in positive_pairs):
            raise ValueError("All positive pairs must contain at least 2 values")
        if not all(len(pair) >= 2 for pair in negative_pairs):
            raise ValueError("All negative pairs must contain at least 2 values")
            
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        drug_pos_id, target_pos_id = self.positive_pairs[idx][:2]  # Take first 2 values
        
        if idx < len(self.negative_pairs):
            pair = self.negative_pairs[idx]
            if len(pair) > 2:
                # warnings.warn(f"Negative pair at index {idx} has {len(pair)} values, using only first 2")
                drug_neg_id, target_neg_id = pair[:2]  # Take first 2 values
        else:
            drug_neg_id, target_neg_id = 0, 0

        return {
            "drug_pos_id": torch.tensor(drug_pos_id, dtype=torch.long),
            "target_pos_id": torch.tensor(target_pos_id, dtype=torch.long),
            "drug_neg_id": torch.tensor(drug_neg_id, dtype=torch.long), 
            "target_neg_id": torch.tensor(target_neg_id, dtype=torch.long),
            "label_pos": torch.tensor(1, dtype=torch.long),  # Positive label
            "label_neg": torch.tensor(0, dtype=torch.long)
        }
def load_data(adjacency_matrix, feature_matrix, positive_samples, AY, batch_size=32, test_split=0.2):
    num_neg_samples = len(positive_samples) * 3
    negative_samples = generate_negative_samples(AY, num_neg_samples)

    pos_train, pos_val = train_test_split(positive_samples, test_size=test_split, random_state=42)
    neg_train, neg_val = train_test_split(negative_samples, test_size=test_split, random_state=42)

    pos_train = pos_train * 2

    train_dataset = DTIDataset(pos_train, neg_train)
    val_dataset = DTIDataset(pos_val, neg_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pyg_graph = create_pyg_graph(adjacency_matrix, feature_matrix)

    return train_loader, val_loader, pyg_graph

def generate_negative_samples(AY, num_neg_samples):
    """
    Generates negative samples by selecting (drug, target) pairs where AY == 0.
    """
    nd, nt = AY.shape
    neg_samples = []
    
    while len(neg_samples) < num_neg_samples:
        d = np.random.randint(0, nd)  # Random drug index
        t = np.random.randint(0, nt)  # Random target index
        if AY[d, t] == 0:  # Ensure it's not a positive sample
            neg_samples.append((d, t, 0))  # Label 0 for negative sample

    return neg_samples