from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm.auto import tqdm

from models.model import DTIBioBERT
from utils.utils import DTIDataset, get_device, compute_class_weights, compute_metrics
from evaluate import evaluate_model

def train_model(train_csv, val_csv, model_save_path, epochs=10):
    device = get_device()
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model = DTIBioBERT().to(device)

    train_labels = pd.read_csv(train_csv)['Label']
    class_weights = compute_class_weights(train_csv, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device).to(torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = DTIDataset(train_csv, tokenizer)
    val_dataset = DTIDataset(val_csv, tokenizer)
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    best_loss = 1e8
    counter = 0

    for epoch in range(epochs):
        # Training
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

        metrics = compute_metrics(all_labels, all_probs)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val AUC ROC: {metrics['auc_roc']:.4f}")
        print(f"Val AUC PC: {metrics['auc_pc']:.4f}")
        print(f"Val Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Val Specificity: {metrics['specificity']:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model by validation loss")
        else:
            print("No improvement in validation loss")
            counter += 1

        del batch, input_ids, token_type_ids, attention_mask, labels, outputs, loss
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_csv = "Datasets/train.csv"
    val_csv = "Datasets/val.csv"
    test_csv = "Datasets/test.csv"
    model_save_path = "biobert_dti.pth"

    print("Starting training...")
    train_model(train_csv, val_csv, model_save_path, epochs=50)

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(test_csv, model_save_path)
