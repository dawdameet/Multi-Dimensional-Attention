import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDimTFIDFAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_dims=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_dims = num_dims

        # Attention layers
        self.query = nn.Linear(embed_dim, embed_dim * num_dims)
        self.key = nn.Linear(embed_dim, embed_dim * num_dims)
        self.value = nn.Linear(embed_dim, embed_dim * num_dims)
        
        self.out_proj = nn.Linear(embed_dim * num_dims, embed_dim)

    def forward(self, x, tfidf_scores):
        batch_size, seq_length, embed_dim = x.shape

        # Compute queries, keys, and values for multiple dimensions
        Q = self.query(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        K = self.key(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        V = self.value(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)

        # Compute scaled dot-product attention
        attn_scores = torch.einsum("bqde,bkde->bqkd", Q, K) / (embed_dim ** 0.5)

        # Fix: Expand TF-IDF scores to match attention score dimensions
        tfidf_scores = tfidf_scores.unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, 1, seq_length, 1)
        tfidf_scores = tfidf_scores.expand(batch_size, seq_length, seq_length, self.num_dims)  # Match attn_scores

        # Apply TF-IDF weighting to attention scores
        attn_scores = attn_scores * tfidf_scores  

        # Compute final attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention-weighted values
        context = torch.einsum("bqkd,bkde->bqde", attn_probs, V)
        
        # Flatten dimensions and project back
        context = context.reshape(batch_size, seq_length, -1)
        output = self.out_proj(context)

        return output

batch_size = 2
seq_length = 5
embed_dim = 16
num_heads = 4
num_dims = 2

# Initialize multi-dimensional TF-IDF weighted attention
multi_dim_tfidf_attention = MultiDimTFIDFAttention(embed_dim, num_heads, num_dims)

# Random input (batch_size=2, sequence_length=5, embedding_dim=16)
x = torch.randn(batch_size, seq_length, embed_dim)

# Random TF-IDF scores (values between 0 and 1)
tfidf_scores = torch.rand(batch_size, seq_length)  # Shape: (batch_size, seq_length)

# Pass input through attention with TF-IDF scaling
output = multi_dim_tfidf_attention(x, tfidf_scores)

print("Input Shape:", x.shape)
print("TF-IDF Scores Shape:", tfidf_scores.shape)
print("Output Shape:", output.shape)
