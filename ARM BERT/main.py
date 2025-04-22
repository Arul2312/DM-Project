from utils.utils import set_seed, extract_association_rules, DTIDatasetWithARM
from models.armbert import DTIBioBERTWithARM
from evaluate import evaluate_model_with_arm

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

def train_model_with_arm(train_csv, val_csv, model_save_path, epochs=10, seed=42):
    set_seed(seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Extracting association rules...")
    rules = extract_association_rules(train_csv)
    print(f"Extracted {len(rules)} association rules")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DTIBioBERTWithARM().to(device)
    train_labels = pd.read_csv(train_csv)['Label']
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device).to(torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()
    train_dataset = DTIDatasetWithARM(train_csv, tokenizer, rules_support=rules)
    val_dataset = DTIDatasetWithARM(val_csv, tokenizer, rules_support=rules)
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    best_loss = 1e8
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        # Validation
        model.eval()
        val_loss = 0
        all_labels = []
        all_probs = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()
                val_loss += loss.item()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
        threshold = 0.5
        predictions = (np.array(all_probs) > threshold).astype(int)
        val_acc = accuracy_score(all_labels, predictions)
        val_auc_roc = roc_auc_score(all_labels, all_probs)
        val_auc_pc = average_precision_score(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
        val_sensitivity = tp / (tp + fn)
        val_specificity = tn / (tn + fp)
        scheduler.step(val_loss)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val AUC ROC: {val_auc_roc:.4f}")
        print(f"Val AUC PC: {val_auc_pc:.4f}")
        print(f"Val Sensitivity: {val_sensitivity:.4f}")
        print(f"Val Specificity: {val_specificity:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model by validation loss")
        else:
            print("No improvement in validation loss")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_csv = "Datasets/train.csv"
    val_csv = "Datasets/val.csv"
    test_csv = "Datasets/test.csv"
    model_save_path = "biobert_dti_arm.pth"
    print("Starting training with Association Rule Mining...")
    train_model_with_arm(train_csv, val_csv, model_save_path, epochs=50)
    rules = extract_association_rules(train_csv)
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model_with_arm(test_csv, model_save_path, rules)
