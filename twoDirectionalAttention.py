import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoDirectionalMultiDimAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_dims=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_dims = num_dims
        self.query_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.key_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.value_fwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.query_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.key_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.value_bwd = nn.Linear(embed_dim, embed_dim * num_dims)
        self.out_proj = nn.Linear(embed_dim * num_dims, embed_dim)

    def forward(self, x, tfidf_scores):
        batch_size, seq_length, embed_dim = x.shape
        Q_fwd = self.query_fwd(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        K_fwd = self.key_fwd(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        V_fwd = self.value_fwd(x).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        attn_scores_fwd = torch.einsum("bqde,bkde->bqkd", Q_fwd, K_fwd) / (embed_dim ** 0.5)
        x_reversed = torch.flip(x, [1])
        Q_bwd = self.query_bwd(x_reversed).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        K_bwd = self.key_bwd(x_reversed).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        V_bwd = self.value_bwd(x_reversed).reshape(batch_size, seq_length, self.num_dims, self.embed_dim)
        attn_scores_bwd = torch.einsum("bqde,bkde->bqkd", Q_bwd, K_bwd) / (embed_dim ** 0.5)
        tfidf_scores = tfidf_scores.unsqueeze(1).unsqueeze(-1).expand(batch_size, seq_length, seq_length, self.num_dims)
        attn_scores_fwd = attn_scores_fwd * tfidf_scores
        attn_scores_bwd = attn_scores_bwd * tfidf_scores
        attn_probs_fwd = F.softmax(attn_scores_fwd, dim=-1)
        attn_probs_bwd = F.softmax(attn_scores_bwd, dim=-1)
        context_fwd = torch.einsum("bqkd,bkde->bqde", attn_probs_fwd, V_fwd)
        context_bwd = torch.einsum("bqkd,bkde->bqde", attn_probs_bwd, V_bwd)
        alpha = 0.5
        context = alpha * context_fwd + (1 - alpha) * context_bwd
        context = context.reshape(batch_size, seq_length, -1)
        output = self.out_proj(context)
        return output



batch_size = 2
seq_length = 5
embed_dim = 16
num_heads = 4
num_dims = 2


two_directional_attention = TwoDirectionalMultiDimAttention(embed_dim, num_heads, num_dims)

x = torch.randn(batch_size, seq_length, embed_dim)

tfidf_scores = torch.rand(batch_size, seq_length)

output = two_directional_attention(x, tfidf_scores)

print("Input Shape:", x.shape)
print("TF-IDF Scores Shape:", tfidf_scores.shape)
print("Output Shape:", output.shape)
