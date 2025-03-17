import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

dataset = load_dataset("imdb")

train_texts = [x['text'] for x in dataset['train']]
train_labels = torch.tensor([x['label'] for x in dataset['train']])

test_texts = [x['text'] for x in dataset['test']]
test_labels = torch.tensor([x['label'] for x in dataset['test']])

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(train_texts)

train_tfidf = vectorizer.transform(train_texts).toarray()
test_tfidf = vectorizer.transform(test_texts).toarray()

train_tfidf = torch.tensor(train_tfidf, dtype=torch.float32)
test_tfidf = torch.tensor(test_tfidf, dtype=torch.float32)

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
        tfidf_scores = tfidf_scores.unsqueeze(-1)  # Shape: [batch, seq_len, 1]
        attn_scores_fwd = attn_scores_fwd * tfidf_scores
        attn_scores_bwd = attn_scores_bwd * tfidf_scores
        attn_probs_fwd = F.softmax(attn_scores_fwd, dim=-2)
        attn_probs_bwd = F.softmax(attn_scores_bwd, dim=-2)
        context_fwd = torch.einsum("bqkd,bkde->bqde", attn_probs_fwd, V_fwd)
        context_bwd = torch.einsum("bqkd,bkde->bqde", attn_probs_bwd, V_bwd)
        alpha = 0.5
        context = alpha * context_fwd + (1 - alpha) * context_bwd
        context = context.reshape(batch_size, seq_length, -1)
        return self.out_proj(context)

class TransformerWithTFIDF(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)  
        self.attention = TwoDirectionalMultiDimAttention(embed_dim, num_heads, num_dims=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, tfidf_scores):
        x = x.unsqueeze(-1) 
        x = self.embedding(x)  
        seq_length = x.shape[1]
        tfidf_scores = tfidf_scores.unsqueeze(-1) 
        x = self.attention(x, tfidf_scores)
        x = self.encoder(x)
        x = x.mean(dim=1)  
        return self.fc(x)

embed_dim = 16
num_heads = 4
num_layers = 2
num_classes = 2

model = TransformerWithTFIDF(embed_dim, num_heads, num_layers, num_classes)

train_dataset = TensorDataset(train_tfidf, train_labels)
test_dataset = TensorDataset(test_tfidf, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for tfidf_scores, labels in train_loader:
        tfidf_scores, labels = tfidf_scores.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(tfidf_scores, tfidf_scores)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for tfidf_scores, labels in test_loader:
        tfidf_scores = tfidf_scores.to(device)
        outputs = model(tfidf_scores, tfidf_scores)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)
        actuals.extend(labels.numpy())

accuracy = accuracy_score(actuals, predictions)
print(f"Test Accuracy: {accuracy:.4f}")