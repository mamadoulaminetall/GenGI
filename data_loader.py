"""
GenGI — Data Loader
ClinVar SNV download, parsing, Ensembl sequence fetching, dataset preparation.
"""

from __future__ import annotations

import gzip
import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic"}
BENIGN_LABELS = {"Benign", "Likely benign"}

ENSEMBL_BASE = "https://rest.ensembl.org"


# ---------------------------------------------------------------------------
# ClinVar download & parsing
# ---------------------------------------------------------------------------


def download_clinvar(save_dir: str = "data/") -> Path:
    """
    Download ClinVar variant_summary.txt.gz if not already cached.

    Parameters
    ----------
    save_dir : str
        Directory to save the file.

    Returns
    -------
    path : Path
        Local path to the downloaded file.
    """
    save_path = Path(save_dir) / "variant_summary.txt.gz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"ClinVar already cached at {save_path}")
        return save_path

    print(f"Downloading ClinVar from {CLINVAR_URL} ...")
    response = requests.get(CLINVAR_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="ClinVar"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved to {save_path}")
    return save_path


def load_clinvar_snvs(path: str | Path, n: int = 5000) -> pd.DataFrame:
    """
    Load and filter ClinVar to balanced SNV pathogenic/benign set.

    Parameters
    ----------
    path : str | Path
        Path to variant_summary.txt.gz.
    n : int
        Total number of variants to return (n/2 per class).

    Returns
    -------
    df : pd.DataFrame
        Columns: GeneSymbol, Chromosome, Start, ReferenceAllele, AlternateAllele, label
    """
    print("Parsing ClinVar variants ...")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)

    # Keep required columns only
    cols = [
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
        "ClinicalSignificance",
        "Assembly",
    ]
    df = df[cols].copy()

    # GRCh38 only
    df = df[df["Assembly"] == "GRCh38"]

    # SNVs: single nucleotide substitutions
    df = df[
        df["ReferenceAllele"].str.len() == 1
        & df["AlternateAllele"].str.len() == 1
        & df["ReferenceAllele"].str.match(r"^[ACGTacgt]$")
        & df["AlternateAllele"].str.match(r"^[ACGTacgt]$")
    ]
    df["ReferenceAllele"] = df["ReferenceAllele"].str.upper()
    df["AlternateAllele"] = df["AlternateAllele"].str.upper()

    # Exclude same-allele non-variants
    df = df[df["ReferenceAllele"] != df["AlternateAllele"]]

    # Filter by clinical significance
    pathogenic = df[df["ClinicalSignificance"].isin(PATHOGENIC_LABELS)].copy()
    benign = df[df["ClinicalSignificance"].isin(BENIGN_LABELS)].copy()

    pathogenic["label"] = 1
    benign["label"] = 0

    # Balance classes
    n_each = n // 2
    pathogenic = pathogenic.sample(n=min(n_each, len(pathogenic)), random_state=42)
    benign = benign.sample(n=min(n_each, len(benign)), random_state=42)

    combined = pd.concat([pathogenic, benign], ignore_index=True).sample(
        frac=1, random_state=42
    )
    combined = combined.drop(columns=["ClinicalSignificance", "Assembly"])
    combined["Start"] = combined["Start"].astype(int)

    print(
        f"Loaded {len(pathogenic)} pathogenic + {len(benign)} benign SNVs "
        f"(total {len(combined)})"
    )
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ensembl sequence fetching
# ---------------------------------------------------------------------------


def fetch_sequence(
    chrom: str,
    start: int,
    end: int,
    genome: str = "GRCh38",
) -> Optional[str]:
    """
    Fetch a DNA sequence from Ensembl REST API.

    Parameters
    ----------
    chrom : str
        Chromosome name (e.g., '17', 'X').
    start : int
        1-based start position.
    end : int
        1-based end position.
    genome : str
        Genome assembly (currently unused — Ensembl REST defaults to GRCh38).

    Returns
    -------
    sequence : str | None
        Uppercase DNA string, or None on error.
    """
    url = f"{ENSEMBL_BASE}/sequence/region/human/{chrom}:{start}..{end}:1"
    headers = {"Content-Type": "text/plain"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            seq = response.text.strip().upper()
            if seq and all(c in "ACGTN" for c in seq):
                return seq
        return None
    except requests.RequestException:
        return None
    finally:
        time.sleep(0.1)  # Ensembl rate-limit: max ~15 req/s


def get_variant_context(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int = 128,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch ±window bp context around a variant and build ref/alt sequences.

    Parameters
    ----------
    chrom : str
    pos : int   1-based variant position
    ref : str   Reference allele (single nucleotide)
    alt : str   Alternate allele (single nucleotide)
    window : int  Half-window size in bp

    Returns
    -------
    (ref_context, alt_context) : tuple[str | None, str | None]
        Full context sequences with ref / alt at the central position.
        Both are None if the Ensembl fetch fails.
    """
    start = max(1, pos - window)
    end = pos + window

    seq = fetch_sequence(chrom, start, end)
    if seq is None:
        return None, None

    # Determine where the variant sits within the fetched sequence
    var_offset = pos - start  # 0-based index in seq

    if var_offset < 0 or var_offset >= len(seq):
        return None, None

    # Verify reference matches
    if seq[var_offset].upper() != ref.upper():
        # Mismatch — still proceed but replace the centre with ref/alt
        pass

    ref_context = seq[:var_offset] + ref.upper() + seq[var_offset + 1 :]
    alt_context = seq[:var_offset] + alt.upper() + seq[var_offset + 1 :]

    return ref_context, alt_context


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    df: pd.DataFrame,
    window: int = 128,
    max_variants: int = 500,
) -> pd.DataFrame:
    """
    Enrich a ClinVar DataFrame with DNA context sequences.

    For each variant, fetches reference and alternate context sequences via
    the Ensembl REST API.  Rows where the fetch fails are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Chromosome, Start, ReferenceAllele, AlternateAllele
    window : int
        Half-window for context fetching (total length ≈ 2*window + 1 bp).
    max_variants : int
        Limit number of variants to process (avoids very long API sessions).

    Returns
    -------
    enriched : pd.DataFrame
        Original columns + ref_seq, alt_seq.  Failed rows are dropped.
    """
    df = df.head(max_variants).copy()
    ref_seqs, alt_seqs = [], []

    print(f"Fetching sequences for {len(df)} variants (window={window}) ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ensembl"):
        ref_ctx, alt_ctx = get_variant_context(
            chrom=str(row["Chromosome"]),
            pos=int(row["Start"]),
            ref=str(row["ReferenceAllele"]),
            alt=str(row["AlternateAllele"]),
            window=window,
        )
        ref_seqs.append(ref_ctx)
        alt_seqs.append(alt_ctx)

    df["sequence_ref"] = ref_seqs
    df["sequence_alt"] = alt_seqs

    # Drop variants where fetching failed
    before = len(df)
    df = df.dropna(subset=["sequence_ref", "sequence_alt"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} variants (Ensembl fetch failed).")

    print(f"Dataset ready: {len(df)} variants.")
    return df.reset_index(drop=True)
