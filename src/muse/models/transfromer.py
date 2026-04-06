from src.muse.models.token_cnn import TokenCNN
import torch.nn as nn
import torch

class MusicTransformer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=2, max_tokens=2048):
        super().__init__()

        self.token_encoder = TokenCNN(embedding_dim)

        self.max_tokens = max_tokens
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_tokens, embedding_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim + 15, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, tokens, token_mask, global_features):
        B, N, C, H, W = tokens.shape

        # flatten tokens
        tokens = tokens.view(B * N, C, H, W)

        # encode tokens
        embeddings = self.token_encoder(tokens)  # [B*N, D]
        embeddings = embeddings.view(B, N, -1)   # [B, N, D]
        embeddings = embeddings + self.pos_embedding[:, :N, :]

        # Transformer (masking)
        src_key_padding_mask = ~token_mask  # True = ignore
        x = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        # masked mean pooling
        mask = token_mask.unsqueeze(-1)
        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # concat global features
        x = torch.cat([x, global_features], dim=1)

        return self.head(x)