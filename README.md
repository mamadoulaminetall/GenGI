# 🧬 GenGI — General Genome Interpretation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)

## Abstract

**GenGI** is a research proof-of-concept that couples the **Nucleotide Transformer v2** (InstaDeepAI/EMBL-EBI, 50M parameters) with a custom **PyTorch TransformerEncoder** to predict variant pathogenicity from raw whole-exome sequencing (WES) variant calls. Unlike annotation-based tools (CADD, REVEL), GenGI operates directly on DNA sequence: for each variant `chr:pos:ref:alt`, it fetches ±128 bp of genomic context via the Ensembl REST API, computes two CLS embeddings (reference and alternate), and feeds the *mutation-effect vector* Δ = embed(alt) − embed(ref) into a pre-LayerNorm TransformerEncoder. A learnable CLS token aggregates information across all variants of a patient, enabling joint, context-aware pathogenicity scoring. Gradient × input attribution provides single-variant explainability. The model is trained on balanced ClinVar SNVs (Pathogenic / Benign, GRCh38) and evaluated by AUROC.

## Architecture

```
WES variants  chr:pos:ref:alt
      │
      ▼
Ensembl REST API  →  ±128 bp DNA context
      │                     │
  ref_seq               alt_seq
      │                     │
      ▼                     ▼
┌──────────────────────────────────┐
│  Nucleotide Transformer v2       │  InstaDeepAI (50M params)
│  6-mer tokenisation              │
│  CLS embedding  [D = 512]        │
└──────────────────────────────────┘
      │                     │
  embed(ref)          embed(alt)
      └──────────┬──────────┘
                 ▼
       Δ = embed(alt) − embed(ref)   [512-dim mutation-effect vector]
                 │
                 ▼
┌──────────────────────────────────┐
│  GenGI TransformerEncoder        │
│  Linear(512 → 256)               │
│  [CLS] prepended                 │
│  TransformerEncoder (2L, 4H)     │
│  Pre-LN, GELU, d_ff = 1024       │
└──────────────────────────────────┘
                 │
                 ▼
LayerNorm → Linear → GELU → Dropout → Linear(1)
                 │
                 ▼
     Pathogenicity score  ∈ [0, 1]

XAI:  gradient × Δ  →  per-variant importance score
```

## Quick Start

```bash
git clone https://github.com/mlaminetall/GenGI
cd GenGI
pip install -r requirements.txt

# Demo mode (no model download, mock embeddings, instant startup)
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select **Demo mode** in the sidebar to explore the full UI with mock embeddings.

## Training

```bash
# Fast demo training (random embeddings, no Ensembl API)
python train.py --mock-embedder --epochs 10 --n-variants 1000

# Full pipeline (downloads ClinVar ~300 MB, calls Ensembl REST, loads NT model)
python train.py --epochs 20 --batch-size 32 --device cpu

# GPU training
python train.py --epochs 30 --batch-size 64 --device cuda
```

Training logs: `train_loss`, `val_loss`, `val_AUROC` per epoch. Best checkpoint saved to `models/gengi_best.pt`.

## Scientific Context

**Nucleotide Transformer (NT)**  
Dalla-Torre et al. (2023) introduced NT, a family of DNA foundation models pre-trained on 2,500+ reference genomes and the human reference (GRCh38). The v2 variants use species-aware multi-species pre-training and achieve state-of-the-art on 18 downstream genomics tasks (histone marks, splice sites, regulatory elements). We use the 50M multi-species variant as a frozen encoder.
> Dalla-Torre H. et al. *The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics.* bioRxiv (2023). https://doi.org/10.1101/2023.01.11.523679

**DNABERT-2**  
Ji et al. / Zhou et al. demonstrated that BERT-style pre-training on DNA with Byte Pair Encoding improves generalisation across species, motivating the use of large pre-trained DNA LLMs as universal feature extractors.
> Zhou Z. et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* arXiv:2306.15006 (2023).

**ClinVar**  
The training data derives from NCBI ClinVar, the most comprehensive public repository of human genomic variants and their clinical significance. We filter to GRCh38 SNVs with high-confidence Pathogenic / Benign classifications.
> Landrum M.J. et al. *ClinVar: improving access to variant interpretations and supporting evidence.* NAR (2018). https://doi.org/10.1093/nar/gkx1153

**UK Biobank WES**  
The long-term target application is the UK Biobank whole-exome sequencing cohort (200,000+ participants), enabling phenotype-level variant burden analysis.
> Backman J.D. et al. *Exome sequencing and analysis of 454,787 UK Biobank participants.* Nature (2021). https://doi.org/10.1038/s41586-021-04103-z

**GenomicBERT / GENA-LM**  
Concurrent work confirms that transformer-based DNA encoders transfer effectively to clinical variant interpretation tasks, supporting the GenGI design.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ POC | ClinVar SNV binary classification (Pathogenic/Benign) using NT delta embeddings + GenGI |
| **Phase 2** | Planned | UK Biobank WES — multi-patient cohort, phenotype association, larger variant sets per patient |
| **Phase 3** | Planned | Multi-phenotype structured output, cross-modal integration (gene expression, protein structure) |
| **Phase 4** | Planned | Clinical integration API, ACMG criteria alignment, ensemble with CADD/AlphaMissense |

## Project Structure

```
GenGI/
├── app.py              # Streamlit demo application
├── model.py            # GenGI PyTorch model + VariantDataset
├── embedder.py         # DNAEmbedder (Nucleotide Transformer) + MockEmbedder
├── data_loader.py      # ClinVar download, Ensembl sequence fetching, dataset prep
├── train.py            # Training script with AdamW + cosine LR
├── requirements.txt
├── .gitignore
├── data/               # ClinVar cache (gitignored)
└── models/             # Saved checkpoints (gitignored)
```

## Author

**Dr. Mamadou Lamine TALL**  
PhD Bioinformatics · Founder, MedFlow AI  
Contact: [laminetall30@gmail.com](mailto:laminetall30@gmail.com)

*Inspired by the [AI4GI group](https://www.igmm.cnrs.fr), IGMM CNRS Montpellier (Dr. Daniele Raimondi) and their work on AI for genome interpretation.*
