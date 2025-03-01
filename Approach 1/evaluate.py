from src.utils.utils import *
from src.models.mlgann import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, test_loader, graph, adjacency_matrix, device="mps"):
    model.eval()  # Set model to evaluation mode
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)

    all_labels = []
    all_predictions = []
    all_scores = []

    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)  # Progress bar

    with torch.no_grad():  # Disable gradient tracking
        for batch in progress_bar:
            # Extract positive and negative pair indices
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)
            label_pos = batch["label_pos"].cpu().numpy()
            label_neg = batch["label_neg"].cpu().numpy()

            # Forward pass with adjacency matrix
            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, drug_neg_ids, target_neg_ids, adjacency_matrix)
            pos_score = torch.sigmoid(torch.sum(zD_pos * zT_pos, dim=1))
            neg_score = torch.sigmoid(torch.sum(zD_neg * zT_neg, dim=1))
            scores = torch.cat([pos_score, neg_score])
            labels = np.concatenate([label_pos, label_neg])

            # Collect scores and labels
            scores_np = scores.cpu().numpy()
            all_scores.extend(scores_np)
            all_labels.extend(labels)

            # Update progress bar with batch info
            progress_bar.set_postfix(current_batch_size=len(labels))

        # Dynamic threshold based on mean score
        threshold = np.mean(all_scores)
        all_predictions = [1.0 if score > threshold else 0.0 for score in all_scores]

    # Compute evaluation metrics
    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions, zero_division=0)
    rec = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)

    # Print final evaluation metrics
    print("\nModel Evaluation Results:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Target Labels (last 10): {all_labels[-10:]}")
    print(f"Predictions (last 10): {all_predictions[-10:]}")

    return acc, prec, rec, f1, auc_roc, auprc
