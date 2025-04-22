import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from tqdm import tqdm
from models.armbert import DTIBioBERTWithARM
from utils.utils import DTIDatasetWithARM

def evaluate_model_with_arm(test_csv, model_path, rules=None):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DTIBioBERTWithARM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_dataset = DTIDatasetWithARM(test_csv, tokenizer, rules_support=rules)
    test_loader = DataLoader(test_dataset, batch_size=64 if torch.cuda.is_available() else 32, num_workers=4, pin_memory=True)
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
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    predictions = (np.array(all_probs) > optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    metrics = {
        'accuracy': accuracy_score(all_labels, predictions),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pc': average_precision_score(all_labels, all_probs),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'optimal_threshold': optimal_threshold
    }
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC ROC: {metrics['auc_roc']:.4f}")
    print(f"Test AUC PC: {metrics['auc_pc']:.4f}")
    print(f"Test Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {metrics['specificity']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    return metrics
