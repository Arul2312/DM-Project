from src.utils.utils import *
from src.models.mlgann import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve

def evaluate_model(model, test_loader, graph, adjacency_matrix, device="mps"):
    model.eval()
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)

    all_labels = []
    all_predictions = []
    all_scores = []
    pos_scores_all = []
    neg_scores_all = []

    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            # Extract batch data
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)
            label_pos = batch["label_pos"].cpu().numpy()
            label_neg = batch["label_neg"].cpu().numpy()

            # Get embeddings
            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, 
                                                  drug_neg_ids, target_neg_ids, adjacency_matrix)
            
            # Normalize embeddings
            zD_pos = F.normalize(zD_pos, p=2, dim=1)
            zT_pos = F.normalize(zT_pos, p=2, dim=1)
            zD_neg = F.normalize(zD_neg, p=2, dim=1)
            zT_neg = F.normalize(zT_neg, p=2, dim=1)

            # Calculate similarity scores with temperature scaling
            temperature = 0.1
            pos_score = torch.sum(zD_pos * zT_pos, dim=1) / temperature
            neg_score = torch.sum(zD_neg * zT_neg, dim=1) / temperature
            
            # Collect scores and labels
            scores = torch.cat([pos_score, neg_score]).cpu().numpy()
            labels = np.concatenate([label_pos, label_neg])
            
            pos_scores_all.extend(pos_score.cpu().numpy())
            neg_scores_all.extend(neg_score.cpu().numpy())
            all_scores.extend(scores)
            all_labels.extend(labels)

            progress_bar.set_postfix(current_batch_size=len(labels))

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    pos_scores_all = np.array(pos_scores_all)
    neg_scores_all = np.array(neg_scores_all)

    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    
    # Make predictions
    all_predictions = (all_scores > threshold).astype(float)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_scores),
        'auprc': average_precision_score(all_labels, all_scores)
    }

    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"AUPRC:     {metrics['auprc']:.4f}")
    
    print("\nScore Distribution:")
    print(f"Positive Scores - Mean: {np.mean(pos_scores_all):.4f}, Std: {np.std(pos_scores_all):.4f}")
    print(f"Negative Scores - Mean: {np.mean(neg_scores_all):.4f}, Std: {np.std(neg_scores_all):.4f}")
    print(f"Score Separation: {np.mean(pos_scores_all) - np.mean(neg_scores_all):.4f}")

    return tuple(metrics.values())
