import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.muse.dataset import MuSeDataset
from src.muse.models.transfromer import MusicTransformer
from src.muse.collate import collate_fn

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

df = pd.read_csv("datasets\\muse\\raw\\muse_v3.csv")
audio_root = "datasets\\muse\\audio"


dataset = MuSeDataset(df=df, audio_root=audio_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = MusicTransformer(
    embedding_dim=128,
    num_heads=4,
    num_layers=2,
    max_tokens=2048
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        if batch is None:
            continue

        tokens = batch["tokens"].to(device)                 # [B, N, 1, 128, 45]
        token_mask = batch["token_mask"].to(device)         # [B, N]
        global_features = batch["global_features"].to(device)  # [B, 15]
        vad = batch["vad"].to(device)                       # [B, 3]

        optimizer.zero_grad()

        preds = model(tokens, token_mask, global_features)  # [B, 3]
        loss = criterion(preds, vad)

        loss.backward()
        optimizer.step()

        batch_size = vad.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}")