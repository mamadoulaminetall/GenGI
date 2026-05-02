"""
GenGI — Training Script
ClinVar SNV pathogenicity classification using delta embeddings + TransformerEncoder.

Usage
-----
# Demo mode (no API calls, random embeddings):
python train.py --mock-embedder --epochs 5

# Full pipeline (downloads ClinVar, calls Ensembl REST API):
python train.py --epochs 20 --batch-size 32 --device cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, classification_report

from data_loader import download_clinvar, load_clinvar_snvs, prepare_dataset
from embedder import DNAEmbedder, MockEmbedder
from model import GenGI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def embed_dataframe(df, embedder, batch_size: int = 8) -> torch.Tensor:
    """Return delta embeddings [N, D] for all rows in df."""
    ref_seqs = df["sequence_ref"].tolist()
    alt_seqs = df["sequence_alt"].tolist()
    deltas = []
    for i in range(0, len(ref_seqs), batch_size):
        delta = embedder.embed_variant_delta(
            ref_seqs[i : i + batch_size],
            alt_seqs[i : i + batch_size],
        )
        deltas.append(delta.cpu())
    return torch.cat(deltas, dim=0)  # [N, D]


def build_tensor_dataset(deltas: torch.Tensor, labels: torch.Tensor) -> TensorDataset:
    """Wrap pre-computed embeddings into a TensorDataset."""
    # Each sample: delta [1, D], label scalar
    return TensorDataset(deltas.unsqueeze(1), labels.float())  # [N, 1, D], [N]


def evaluate(model: GenGI, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    loss = nn.BCEWithLogitsLoss()(logits, labels).item()
    try:
        auroc = roc_auc_score(labels_np, probs)
    except ValueError:
        auroc = float("nan")
    return loss, auroc, probs, labels_np


# ---------------------------------------------------------------------------
# Mock dataset (when --mock-embedder is set, skip Ensembl calls)
# ---------------------------------------------------------------------------


def build_mock_dataset(n: int = 1000, embed_dim: int = 512):
    """Generate random embeddings + balanced binary labels for demo training."""
    rng = np.random.RandomState(42)
    # Simulate: pathogenic variants cluster around +0.3, benign around -0.3
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32)
    np.random.shuffle(labels)
    center = np.where(labels == 1, 0.3, -0.3)[:, None]
    deltas = rng.randn(n, embed_dim).astype(np.float32) + center
    return torch.from_numpy(deltas), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train GenGI variant classifier")
    parser.add_argument("--data-dir", default="data/", help="Directory for ClinVar cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--mock-embedder",
        action="store_true",
        help="Use random embeddings (no Ensembl API calls, no NT model download)",
    )
    parser.add_argument(
        "--n-variants",
        type=int,
        default=1000,
        help="Number of ClinVar variants to use (per class: n/2)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs("models", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if args.mock_embedder:
        print("Mock mode: generating synthetic embeddings (no API calls).")
        deltas, labels = build_mock_dataset(n=args.n_variants)
        embed_dim = 512
    else:
        clinvar_path = download_clinvar(save_dir=args.data_dir)
        df = load_clinvar_snvs(clinvar_path, n=args.n_variants)
        df = prepare_dataset(df, window=128, max_variants=args.n_variants)

        print("Loading Nucleotide Transformer ...")
        embedder = DNAEmbedder(device=args.device)
        embed_dim = DNAEmbedder.EMBED_DIM

        print("Embedding variants ...")
        deltas = embed_dataframe(df, embedder, batch_size=8)
        labels = torch.tensor(df["label"].values, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 2. Train / val / test split (80 / 10 / 10)
    # ------------------------------------------------------------------
    n_total = len(labels)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    dataset = build_tensor_dataset(deltas, labels)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # ------------------------------------------------------------------
    # 3. Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = GenGI(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"GenGI parameters: {model.count_parameters():,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_val_auroc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        train_loss = np.mean(train_losses)
        val_loss, val_auroc, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auroc": val_auroc,
                    "args": vars(args),
                },
                "models/gengi_best.pt",
            )

    print(f"\nBest model: epoch {best_epoch}, val AUROC={best_val_auroc:.4f}")
    print("Saved to models/gengi_best.pt")

    # ------------------------------------------------------------------
    # 5. Final test evaluation
    # ------------------------------------------------------------------
    checkpoint = torch.load("models/gengi_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_auroc, probs, labels_np = evaluate(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    print(f"\nTest AUROC: {test_auroc:.4f}")
    print(f"Test loss:  {test_loss:.4f}")
    print("\nClassification report:")
    print(classification_report(labels_np.astype(int), preds, target_names=["Benign", "Pathogenic"]))


if __name__ == "__main__":
    main()
