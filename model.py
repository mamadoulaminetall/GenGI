"""
GenGI — General Genome Interpretation
PyTorch model: TransformerEncoder over variant delta-embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional


class GenGI(nn.Module):
    """
    General Genome Interpretation model.

    Input : delta embeddings  [B, N_variants, D]  (alt_embed - ref_embed)
    Output: pathogenicity logits  [B, 1]

    Architecture
    ------------
    1. Project input_dim → hidden_dim
    2. Prepend learnable CLS token
    3. TransformerEncoder (Pre-LN, GELU, 2 layers, 4 heads)
    4. Classify CLS output: LayerNorm → Linear → GELU → Dropout → Linear(1)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project LLM embedding dimension → model hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable CLS token  [1, 1, hidden_dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder with Pre-LayerNorm for training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            norm_first=True,       # Pre-LN
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Pre-LN does not support nested tensors
        )

        # Classification head applied to the CLS token output
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        delta_embeddings: torch.Tensor,          # [B, N, D]
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, N+1] bool
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        delta_embeddings : [B, N, D]
            Mutation-effect vectors (alt_embed - ref_embed) per variant.
        src_key_padding_mask : [B, N+1], optional
            True where positions should be ignored (padding).

        Returns
        -------
        logits : [B, 1]
        """
        B, N, _ = delta_embeddings.shape

        # Project input → hidden_dim  [B, N, hidden_dim]
        x = self.input_proj(delta_embeddings)

        # Expand CLS token for each item in the batch  [B, 1, hidden_dim]
        cls = self.cls_token.expand(B, -1, -1)

        # Prepend CLS  [B, N+1, hidden_dim]
        x = torch.cat([cls, x], dim=1)

        # Run TransformerEncoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract CLS output  [B, hidden_dim]
        cls_out = x[:, 0, :]

        # Classify  [B, 1]
        logits = self.classifier(cls_out)
        return logits

    def get_variant_importance(
        self,
        delta_embeddings: torch.Tensor,   # [1, N, D]  — single patient
    ) -> np.ndarray:
        """
        Gradient × input attribution for each variant.

        Returns
        -------
        importance : np.ndarray [N]
            Scalar importance score per variant (higher = more influential).
        """
        delta_embeddings = delta_embeddings.requires_grad_(True)

        logits = self.forward(delta_embeddings)              # [1, 1]
        score = torch.sigmoid(logits).squeeze()             # scalar
        score.backward()

        # Gradient × input magnitude averaged over embedding dimension
        grad = delta_embeddings.grad                        # [1, N, D]
        importance = (grad * delta_embeddings).abs().mean(dim=-1).squeeze(0)  # [N]
        return importance.detach().cpu().numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VariantDataset(Dataset):
    """
    PyTorch Dataset for variant pathogenicity prediction.

    Expects a DataFrame with columns:
        - sequence_ref : str  (reference context sequence)
        - sequence_alt : str  (alternate context sequence)
        - label        : int  (0=benign, 1=pathogenic)

    The embedder is applied at construction time; delta embeddings are stored
    in memory as float32 tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embedder,
        batch_size: int = 8,
    ):
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

        ref_seqs = df["sequence_ref"].tolist()
        alt_seqs = df["sequence_alt"].tolist()

        # Embed in batches, concatenate results
        deltas = []
        for i in range(0, len(ref_seqs), batch_size):
            ref_batch = ref_seqs[i : i + batch_size]
            alt_batch = alt_seqs[i : i + batch_size]
            delta = embedder.embed_variant_delta(ref_batch, alt_batch)  # [k, D]
            deltas.append(delta.cpu())

        # Each sample is a single variant; wrap as [1, D] for GenGI's [B, N, D] contract
        self.deltas = torch.cat(deltas, dim=0)  # [total, D]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Return [1, D] delta and scalar label
        return self.deltas[idx].unsqueeze(0), self.labels[idx]
