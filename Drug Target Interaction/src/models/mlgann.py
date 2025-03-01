import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLGANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads=4, dropout=0.2):
        super(MLGANN, self).__init__()

        self.gcns = nn.ModuleList([
        GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
        for i in range(num_layers)
        ])

        for gcn in self.gcns:
            nn.init.xavier_normal_(gcn.lin.weight, gain=1.0)  # Corrected to gcn.lin.weight
            if gcn.bias is not None:  # Bias is directly accessible, but typically None in GCNConv
                nn.init.zeros_(gcn.bias)

        self.W_D = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_T = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.kaiming_normal_(self.W_D.weight)
        nn.init.kaiming_normal_(self.W_T.weight)

        self.q_D = nn.Parameter(torch.randn(hidden_dim))
        self.q_T = nn.Parameter(torch.randn(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, drug_pos_ids, target_pos_ids, drug_neg_ids, target_neg_ids, adjacency_matrix):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.shape[0]
        adjacency_matrix = adjacency_matrix[:num_nodes, :num_nodes]

        layer_embeddings = []
        # print(edge_index)
        for gcn in self.gcns:
            x = F.relu(gcn(x, edge_index))
            # print(x)
            # if torch.isnan(x).any():
            #     # print("NaN in GCN output")
            #     # print(gcn.weight)
            #     # print(x)
            x = self.bn(x)
            # if torch.isnan(x).any():
            #     # print("NaN after BatchNorm")
            x = self.dropout(x)
            layer_embeddings.append(x)

        layer_embeddings = torch.stack(layer_embeddings, dim=1)
        layer_embeddings = layer_embeddings.permute(1, 0, 2)

        z_D = self.attention_pooling(layer_embeddings, self.W_D, self.q_D)
        # if torch.isnan(z_D).any():
        #     # print("NaN in z_D after attention")
        z_T = self.attention_pooling(layer_embeddings, self.W_T, self.q_T)
        # if torch.isnan(z_T).any():
        #     # print("NaN in z_T after attention")

        # if torch.isnan(z_D).any() or torch.isnan(z_T).any():
            # print("Warning: NaN detected in model output embeddings!")

        z_D = self.output_layer(z_D)
        z_T = self.output_layer(z_T)

    # ... (rest unchanged)

        # # Debug prints
        # print(f"z_D.shape: {z_D.shape}")  # Should be [1722, 64]
        # print(f"drug_pos_ids.shape: {drug_pos_ids.shape}")

        # # Validate indices
        # for ids, name in [(drug_pos_ids, "drug_pos_ids"), (target_pos_ids, "target_pos_ids"),
        #                   (drug_neg_ids, "drug_neg_ids"), (target_neg_ids, "target_neg_ids")]:
        #     if ids.max() >= num_nodes or ids.min() < 0:
        #         print(f"Out of bounds in {name}: max={ids.max()}, min={ids.min()}, num_nodes={num_nodes}")
        #         raise ValueError(f"Index error in {name}")

        drug_emb_pos = z_D[drug_pos_ids]
        target_emb_pos = z_T[target_pos_ids]
        drug_emb_neg = z_D[drug_neg_ids]
        target_emb_neg = z_T[target_neg_ids]

        return drug_emb_pos, target_emb_pos, drug_emb_neg, target_emb_neg

    def attention_pooling(self, embeddings, W, q):
        h = F.leaky_relu(W(embeddings))
        e = torch.matmul(h, q)  # [num_layers, num_nodes]
        alpha = F.softmax(e, dim=0)  # Softmax over layers
        return torch.sum(embeddings * alpha.unsqueeze(-1), dim=0)