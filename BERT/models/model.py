import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class DTIBioBERT(nn.Module):
    def __init__(self, pretrained_model_name='dmis-lab/biobert-v1.1', num_labels=2):
        super().__init__()
        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512
        )
        self.biobert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.biobert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
        return logits
