"""
GenGI — DNA Embedder
Wraps Nucleotide Transformer v2 (InstaDeepAI) for variant embedding.
Also provides MockEmbedder for demo / CI mode.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List


class DNAEmbedder:
    """
    Nucleotide Transformer v2 embedder.

    Produces 512-dimensional CLS embeddings from raw DNA sequences.
    Uses 6-mer tokenisation as expected by the NT v2 model family.

    Parameters
    ----------
    device : str
        'cpu', 'cuda', or 'mps'.
    batch_size : int
        Number of sequences per forward pass (keep low on CPU).
    """

    MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    EMBED_DIM = 512

    def __init__(self, device: str = "cpu", batch_size: int = 8):
        from transformers import AutoTokenizer, AutoModel

        self.device = torch.device(device)
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Embed a list of DNA sequences.

        Parameters
        ----------
        sequences : list[str]
            Raw nucleotide strings (A/C/G/T, upper case recommended).

        Returns
        -------
        embeddings : torch.Tensor [N, 512]
            CLS token embedding for each sequence.
        """
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]

            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens)
            # last_hidden_state: [B, seq_len, hidden]
            # CLS token is at position 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 512]
            all_embeddings.append(cls_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)  # [N, 512]

    @torch.no_grad()
    def embed_variant_delta(
        self,
        ref_seqs: List[str],
        alt_seqs: List[str],
    ) -> torch.Tensor:
        """
        Compute mutation-effect vectors: embed(alt) - embed(ref).

        Parameters
        ----------
        ref_seqs : list[str]
            Reference context sequences.
        alt_seqs : list[str]
            Alternate context sequences (same length, alt allele substituted at centre).

        Returns
        -------
        delta : torch.Tensor [N, 512]
        """
        ref_embed = self.embed(ref_seqs)   # [N, 512]
        alt_embed = self.embed(alt_seqs)   # [N, 512]
        return alt_embed - ref_embed       # [N, 512]


class MockEmbedder:
    """
    Demo-mode embedder that returns random Normal embeddings with correct shape.
    No model weights required — useful for UI demos and unit tests.

    Embeddings are seeded per-sequence (hash of string) so the same sequence
    always yields the same mock vector within a session.
    """

    EMBED_DIM = 512

    def embed(self, sequences: List[str]) -> torch.Tensor:
        """Return random [N, 512] embeddings."""
        N = len(sequences)
        # Use a deterministic seed based on sequence content for reproducibility
        rng = np.random.RandomState(seed=self._seed(sequences))
        embeddings = rng.randn(N, self.EMBED_DIM).astype(np.float32)
        return torch.from_numpy(embeddings)

    def embed_variant_delta(
        self,
        ref_seqs: List[str],
        alt_seqs: List[str],
    ) -> torch.Tensor:
        """Return random delta embeddings [N, 512]."""
        ref_embed = self.embed(ref_seqs)
        alt_embed = self.embed(alt_seqs)
        return alt_embed - ref_embed

    @staticmethod
    def _seed(sequences: List[str]) -> int:
        combined = "".join(sequences)
        return hash(combined) % (2**31)
