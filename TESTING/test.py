import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
dataset = load_dataset("imdb")
vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features for memory

def prepare_data(split):
    texts = [x['text'] for x in dataset[split]]
    labels = torch.tensor([x['label'] for x in dataset[split]], dtype=torch.long)
    tfidf = vectorizer.fit_transform(texts) if split == 'train' else vectorizer.transform(texts)
    return TensorDataset(torch.tensor(tfidf.toarray(), dtype=torch.float32), labels)

train_dataset = prepare_data('train')
test_dataset = prepare_data('test')

# Corrected Model Architecture
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1000, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        # Treat each TF-IDF feature as a token in sequence
        self.embedding = nn.Linear(1, embed_dim)  # Each feature is 1-dimensional
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(embed_dim, 2)  # Final output dimension matches

    def forward(self, x):
        # Reshape input: [batch, features] -> [batch, seq_len=1000, 1]
        x = x.unsqueeze(-1)
        # Embed each feature: [batch, 1000, 1] -> [batch, 1000, embed_dim]
        x = self.embedding(x)
        # Transformer expects [batch, seq_len, embed_dim]
        x = self.transformer(x)
        # Average across sequence dimension
        x = x.mean(dim=1)
        return self.classifier(x)

# DataLoader with proper settings
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2 if device.type == 'cuda' else 0,
    pin_memory=True
)

# Initialize model
model = TransformerClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for x, y in DataLoader(test_dataset, batch_size=256):
        outputs = model(x.to(device))
        predictions.extend(torch.argmax(outputs.cpu(), 1).numpy())
        actuals.extend(y.numpy())

print(f"Test Accuracy: {accuracy_score(actuals, predictions):.4f}")