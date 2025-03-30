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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Drug-Target Interaction Prediction')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension size (default: 256)')
    parser.add_argument('--output_dim', type=int, default=64,
                      help='Output dimension size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                      help='Weight decay (default: 2e-5)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs (default: 200)')
    
    # Scheduler parameters
    parser.add_argument('--patience', type=int, default=8,
                      help='Scheduler patience (default: 8)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--factor', type=float, default=0.5,
                      help='Scheduler reduction factor (default: 0.5)')
    
    # Other settings
    parser.add_argument('--device', type=str, default='mps',
                      help='Device to use (default: mps)')
    parser.add_argument('--feature_dim', type=int, default=128,
                      help='Feature dimension for encodings (default: 128)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, graph, adjacency_matrix, optimizer, scheduler, num_epochs=20, device="mps"):
    model.to(device)
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        it = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, 
                                                  drug_neg_ids, target_neg_ids, adjacency_matrix)
            
            # Normalize embeddings
            zD_pos = F.normalize(zD_pos, p=2, dim=1)
            zT_pos = F.normalize(zT_pos, p=2, dim=1)
            zD_neg = F.normalize(zD_neg, p=2, dim=1)
            zT_neg = F.normalize(zT_neg, p=2, dim=1)
            
            # Calculate loss and backpropagate
            loss = dti_loss(zD_pos, zT_pos, zD_neg, zT_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()
            
            total_loss += loss.item()
            it += 1

        # Validation phase
        val_loss, val_acc = evaluate_on_validation(model, val_loader, graph, adjacency_matrix, dti_loss, device)
        scheduler.step(val_loss)
        
        # Keep original print format
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss / it:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def evaluate_on_validation(model, val_loader, graph, adjacency_matrix, criterion, device="mps"):
    model.eval()
    graph = graph.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    total_loss = 0
    num_batches = 0
    all_labels, all_preds = [], []
    pos_scores_all, neg_scores_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            # Get batch data
            drug_pos_ids = batch["drug_pos_id"].to(device)
            target_pos_ids = batch["target_pos_id"].to(device)
            drug_neg_ids = batch["drug_neg_id"].to(device)
            target_neg_ids = batch["target_neg_id"].to(device)
            label_pos = batch["label_pos"].cpu().numpy()
            label_neg = batch["label_neg"].cpu().numpy()

            # Forward pass
            zD_pos, zT_pos, zD_neg, zT_neg = model(graph, drug_pos_ids, target_pos_ids, 
                                                  drug_neg_ids, target_neg_ids, adjacency_matrix)
            
            # Normalize embeddings
            zD_pos = F.normalize(zD_pos, p=2, dim=1)
            zT_pos = F.normalize(zT_pos, p=2, dim=1)
            zD_neg = F.normalize(zD_neg, p=2, dim=1)
            zT_neg = F.normalize(zT_neg, p=2, dim=1)
            
            # Calculate loss
            loss = criterion(zD_pos, zT_pos, zD_neg, zT_neg)
            total_loss += loss.item()
            num_batches += 1

            # Calculate similarity scores
            pos_score = torch.sum(zD_pos * zT_pos, dim=1)
            neg_score = torch.sum(zD_neg * zT_neg, dim=1)
            
            pos_scores_all.extend(pos_score.cpu().numpy())
            neg_scores_all.extend(neg_score.cpu().numpy())
            
            # Make predictions
            pos_preds = (pos_score > 0.0).float().cpu().numpy()
            neg_preds = (neg_score > 0.0).float().cpu().numpy()
            
            all_preds.extend(pos_preds)
            all_preds.extend(neg_preds)
            all_labels.extend(label_pos)
            all_labels.extend(label_neg)

    avg_loss = total_loss / num_batches
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Keep original validation metric prints
    print("\nValidation Metrics:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Positive Scores - Mean: {np.mean(pos_scores_all):.4f}, Std: {np.std(pos_scores_all):.4f}")
    print(f"Negative Scores - Mean: {np.mean(neg_scores_all):.4f}, Std: {np.std(neg_scores_all):.4f}")
    print(f"Score Separation: {np.mean(pos_scores_all) - np.mean(neg_scores_all):.4f}")
    
    return avg_loss, val_acc

def main():

    args = parse_args()

    target_labels = pd.read_csv("Datasets/raw/target_labels.csv")
    targets = pd.read_csv("Datasets/raw/protein_sequences.csv")["pdb_id"].tolist()
    AY = target_labels.filter(items=targets).to_numpy()
    AM = pd.read_csv("Datasets/processed/AM.csv", index_col=0).to_numpy()

    feature_dim = 128

# Drug features (912 nodes, 304 drugs x 3)
    drug_smiles = pd.read_csv("Datasets/raw/drugbank.csv")['smiles'].tolist()  # Load 304 SMILES
    drug_features_multi = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=feature_dim)
    
    for smi in drug_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)  # Generate Morgan fingerprint
            fp = np.array(fp)
        else:
            fp = np.zeros(feature_dim)
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

    train_loader, val_loader, pyg_graph = load_data(AM, feature_matrix, positive_samples, AY, batch_size=128)
    
    test_loader, _, test_graph = load_data(AM, feature_matrix, positive_samples[:-60], AY)

    input_dim = feature_matrix.shape[1]
    model = MLGANN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    ) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.factor,
        patience=args.patience,
        min_lr=args.min_lr
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        graph=pyg_graph,
        adjacency_matrix=torch.tensor(AM, dtype=torch.float),
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=args.device
    )

    print("\nEvaluating on test set...")

    acc, prec, rec, f1, auc_roc, auprc = evaluate_model(
        trained_model,
        test_loader,
        test_graph,
        torch.tensor(AM, dtype=torch.float),
        device=args.device
    )

if __name__ == "__main__":
    main()