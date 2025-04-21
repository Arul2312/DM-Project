from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

from models.model import DTIBioBERT
from utils.utils import DTIDataset, get_device, compute_metrics

def evaluate_model(test_csv, model_path):
    device = get_device()
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model = DTIBioBERT().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = DTIDataset(test_csv, tokenizer)
    batch_size = 64 if torch.cuda.is_available() else 32
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True
    )

    all_labels = []
    all_probs = []
    test_bar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in test_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            outputs = model(input_ids, token_type_ids, attention_mask)
            probs = torch.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_probs)
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC ROC: {metrics['auc_roc']:.4f}")
    print(f"Test AUC PC: {metrics['auc_pc']:.4f}")
    print(f"Test Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {metrics['specificity']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    return metrics

