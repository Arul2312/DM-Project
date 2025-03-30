import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLGANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads=8, dropout=0.3):
        super(MLGANN, self).__init__()
        
        # Store hidden_dim as class attribute
        self.hidden_dim = hidden_dim
        
        # GCN layers
        self.gcns = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Initialize GCN weights
        for gcn in self.gcns:
            nn.init.xavier_normal_(gcn.lin.weight, gain=1.0)
            if gcn.bias is not None:
                nn.init.zeros_(gcn.bias)

        # Multi-head attention parameters
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections for both drug and target
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def multi_head_attention(self, embeddings):
        batch_size = embeddings.size(1)  # [num_layers, num_nodes, hidden_dim]
        
        # Linear projections
        Q = self.W_Q(embeddings)  
        K = self.W_K(embeddings)
        V = self.W_V(embeddings)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        K = K.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        V = V.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and reshape
        context = context.transpose(0, 2).contiguous()
        context = context.view(-1, batch_size, self.num_heads * self.head_dim)  # Use num_heads * head_dim instead of hidden_dim
        
        return context


    def forward(self, data, drug_pos_ids, target_pos_ids, drug_neg_ids, target_neg_ids, adjacency_matrix):
        x, edge_index = data.x, data.edge_index
        layer_embeddings = []
        current_x = x
        
        # GCN layers with skip connections
        for gcn in self.gcns:
            # GCN operation
            new_x = F.relu(gcn(current_x, edge_index))
            new_x = self.layer_norm(new_x)
            new_x = self.dropout(new_x)
            
            # Skip connection
            current_x = current_x + new_x if current_x.shape == new_x.shape else new_x
            layer_embeddings.append(current_x)

        # Stack and permute layer embeddings [num_layers, num_nodes, hidden_dim]
        layer_embeddings = torch.stack(layer_embeddings, dim=0)
        
        # Multi-head attention for drug and target embeddings
        z = self.multi_head_attention(layer_embeddings)
        z = self.layer_norm(z)
        
        # Split into drug and target embeddings
        z_D = z[-1]  # Take final layer for drugs
        z_T = z[-1]  # Take final layer for targets
        
        # Final projections
        z_D = self.output_layer(z_D)
        z_T = self.output_layer(z_T)

        # Get embeddings for positive and negative samples
        drug_emb_pos = z_D[drug_pos_ids]
        target_emb_pos = z_T[target_pos_ids]
        drug_emb_neg = z_D[drug_neg_ids] 
        target_emb_neg = z_T[target_neg_ids]

        return drug_emb_pos, target_emb_pos, drug_emb_neg, target_emb_neg
            