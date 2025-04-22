import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from mlxtend.frequent_patterns import apriori, association_rules

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_association_rules(train_csv, min_support=0.01, min_confidence=0.6, min_lift=3, max_features=10000, max_len=2):
    train_data = pd.read_csv(train_csv)
    positive_interactions = train_data[train_data['Label'] == 1]

    # Count subsequence frequencies and select top N
    protein_subseqs = []
    drug_subseqs = []
    for _, row in positive_interactions.iterrows():
        protein_seq = row['Target Sequence']
        drug_seq = row['SMILES']
        protein_subseqs += [protein_seq[i:i+3] for i in range(len(protein_seq) - 3)]
        drug_subseqs += [drug_seq[i:i+3] for i in range(len(drug_seq) - 3)]

    # Limit to top-N frequent subsequences
    from collections import Counter
    top_protein = [x for x, _ in Counter(protein_subseqs).most_common(max_features)]
    top_drug = [x for x, _ in Counter(drug_subseqs).most_common(max_features)]

    transaction_matrix = []
    for _, row in positive_interactions.iterrows():
        protein_seq = row['Target Sequence']
        drug_seq = row['SMILES']
        transaction = {}
        for subseq in top_protein:
            transaction[f"P_{subseq}"] = subseq in protein_seq
        for subseq in top_drug:
            transaction[f"D_{subseq}"] = subseq in drug_seq
        transaction_matrix.append(transaction)

    transaction_df = pd.DataFrame(transaction_matrix).astype(bool)

    frequent_itemsets = apriori(
        transaction_df, 
        min_support=min_support, 
        use_colnames=True, 
        max_len=max_len, 
        low_memory=True
    )
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]

    protein_drug_rules = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        protein_antecedents = [item for item in antecedents if item.startswith('P_')]
        drug_antecedents = [item for item in antecedents if item.startswith('D_')]
        protein_consequents = [item for item in consequents if item.startswith('P_')]
        drug_consequents = [item for item in consequents if item.startswith('D_')]
        if (protein_antecedents and drug_consequents) or (drug_antecedents and protein_consequents):
            protein_drug_rules.append({
                'antecedent': antecedents[0][2:] if antecedents[0].startswith(('P_', 'D_')) else antecedents[0],
                'consequent': consequents[0][2:] if consequents[0].startswith(('P_', 'D_')) else consequents[0],
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift']
            })
    return protein_drug_rules

class DTIDatasetWithARM(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512, rules_support=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rules_support = rules_support

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Target Sequence'] + " [SEP] " + row['SMILES']
        if self.rules_support is not None:
            text = self.apply_rules_enhancement(text, row['Label'])
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

    def apply_rules_enhancement(self, text, label):
        if label == 1 and self.rules_support is not None:
            protein_seq, drug_seq = text.split(" [SEP] ")
            for rule in self.rules_support:
                if rule['antecedent'] in protein_seq and rule['consequent'] in drug_seq:
                    drug_seq = drug_seq.replace(rule['consequent'], f"IMPORTANT_{rule['consequent']}")
            text = protein_seq + " [SEP] " + drug_seq
        return text
