from src.utils.utils import *
from src.models.mlgann import *
from evaluate import evaluate_model
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv
from rdkit.Chem import rdFingerprintGenerator
import os

def train_model(model, train_loader, val_loader, graph, adjacency_matrix, optimizer, scheduler, num_epochs=20, device="mps"):
    model.to(device)
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)

            optimizer.zero_grad()
            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, drug_neg_ids, target_neg_ids, adjacency_matrix)
            loss = dti_loss(zD_pos, zT_pos, zD_neg, zT_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc = evaluate_on_validation(model, val_loader, graph, adjacency_matrix, dti_loss, device)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model if it exists
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    return model

def evaluate_on_validation(model, val_loader, graph, adjacency_matrix, criterion, device="mps"):
    model.eval()
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    total_loss = 0
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)
            label_pos = batch["label_pos"].cpu().numpy()
            label_neg = batch["label_neg"].cpu().numpy()

            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, drug_neg_ids, target_neg_ids, adjacency_matrix)
            loss = criterion(zD_pos, zT_pos, zD_neg, zT_neg)
            total_loss += loss.item()

            pos_score = torch.sigmoid(torch.sum(zD_pos * zT_pos, dim=1))
            neg_score = torch.sigmoid(torch.sum(zD_neg * zT_neg, dim=1))
            scores = torch.cat([pos_score, neg_score])
            all_scores.extend(scores.cpu().numpy())
            labels = np.concatenate([label_pos, label_neg])
            all_labels.extend(labels)

            # print(f"Val Pos Score Mean: {pos_score.mean().item():.4f}, Val Neg Score Mean: {neg_score.mean().item():.4f}")
            # print(f"Val Pos Score Last: {pos_score[-1].item()}, Val Neg Score Last: {neg_score[-1].item()}")

        threshold = np.mean(all_scores)  # Dynamic threshold
        all_preds = [1.0 if score > threshold else 0.0 for score in all_scores]

    val_acc = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    print(f"Threshold: {threshold:.4f}")
    print(f"Target Labels (last 10): {all_labels[-10:]}")
    print(f"Predictions (last 10): {all_preds[-10:]}")
    return total_loss / len(val_loader), val_acc

def main():
    target_labels = pd.read_csv("Datasets/raw/target_labels.csv")
    targets = pd.read_csv("Datasets/raw/protein_sequences.csv")["pdb_id"].tolist()
    AY = target_labels.filter(items=targets).to_numpy()
    AM = pd.read_csv("Datasets/processed/AM.csv", index_col=0).to_numpy()

    feature_dim = 128

# Drug features (912 nodes, 304 drugs x 3)
    drug_smiles = pd.read_csv("Datasets/raw/drugbank.csv")['smiles'].tolist()  # Load 304 SMILES
    drug_features_multi = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    
    for smi in drug_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)  # Generate Morgan fingerprint
            fp = np.array(fp)
        else:
            fp = np.zeros(128)
        drug_features_multi.append(fp)
    drug_features_multi = np.array(drug_features_multi)
    print("Drug Features loaded.")

    # Target features (810 nodes, 405 targets x 2)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    sequences = pd.read_csv("Datasets/raw/protein_sequences.csv")["sequence"].tolist()[:405]
    target_features_multi = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()[:feature_dim]
        target_features_multi.append(outputs)
    target_features_multi = np.array(target_features_multi)  # 810 x 128
    print("Target Features loaded.")

    drug_features_multi = (drug_features_multi - np.mean(drug_features_multi, axis=0)) / (np.std(drug_features_multi, axis=0) + 1e-8)
    target_features_multi = (target_features_multi - np.mean(target_features_multi, axis=0)) / (np.std(target_features_multi, axis=0) + 1e-8)
    feature_matrix = np.vstack([drug_features_multi, target_features_multi])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    positive_samples = [(i, j) for i in range(AY.shape[0]) for j in range(AY.shape[1]) if AY[i, j] == 1]
    # print(f"Positive samples: {len(positive_samples)}")
    train_loader, val_loader, pyg_graph = load_data(AM, feature_matrix, positive_samples, AY, batch_size=128)
    # print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    test_loader, _, test_graph = load_data(AM, feature_matrix, positive_samples[:60], AY)
    
    input_dim = feature_matrix.shape[1]
    hidden_dim = 256
    output_dim = 64
    num_layers = 2
    model = MLGANN(input_dim, hidden_dim, output_dim, num_layers, num_heads=4, dropout=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)  # Increase LR slightly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)  # Softer reduction
# In train_model, adjust clipping:
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    trained_model = train_model(model, train_loader, val_loader, pyg_graph, torch.tensor(AM, dtype=torch.float), optimizer, scheduler, num_epochs=20)
    # evaluate_model(model, val_loader, pyg_graph, torch.tensor(AM, dtype=torch.float), dti_loss)
    print("\nEvaluating on test set...")
    acc, prec, rec, f1, auc_roc, auprc = evaluate_model(trained_model, test_loader, test_graph, torch.tensor(AM, dtype=torch.float), device="mps")


if __name__ == "__main__":
    main()