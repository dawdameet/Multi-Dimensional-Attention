import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoDirectionalMultiDimensionalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_dims=2):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.num_dims=num_dims
        self.query_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.key_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.value_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.query_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.key_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.value_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.out_proj = nn.Linear(embed_dim * num_dims, embed_dim)
    def forward(self,x,tfidf_scores):
        bat_size,seq_length,embed_dim=x.shape
        Q_fwd=self.query_fwd(x).reshape(bat_size, seq_length, self.num_dims, self.embed_dim)
        