# 🧬 GenGI — General Genome Interpretation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Nucleotide%20Transformer-FFD21E)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

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

### Requirements
- Python 3.10+
- (Optional) GPU for the full Nucleotide Transformer backbone

```bash
git clone https://github.com/mamadoulaminetall/GenGI.git
cd GenGI
pip install -r requirements.txt
```

### Demo mode — instant startup, no downloads

```bash
streamlit run app.py
# → http://localhost:8501
```

The app starts in **Demo mode** (deterministic `MockEmbedder`) — full UI, no model download.
Use the **Try example** button to score 5 ClinVar variants (BRCA1, BRCA2, TP53) immediately.

### Full model (Nucleotide Transformer, ~200 MB)

```bash
pip install transformers accelerate
streamlit run app.py
# Switch to "Full model" in the sidebar
```

## Training on ClinVar

```bash
# Fast mock training — no API calls, ~2 min on CPU
python train.py --mock-embedder --epochs 20 --n-variants 2000

# Full pipeline — downloads ClinVar (~300 MB), fetches Ensembl sequences, loads NT model
python train.py --epochs 30 --n-variants 5000 --batch-size 32

# GPU training
python train.py --device cuda --epochs 30 --n-variants 10000 --batch-size 64
```

Training logs: `train_loss`, `val_loss`, `val_AUROC` per epoch. Best checkpoint saved to `models/gengi_best.pt`.

### Expected performance (ClinVar pathogenic vs benign SNVs)

| Encoder | Val AUROC |
|---------|-----------|
| MockEmbedder (random baseline) | ~0.50 |
| NT-v2-50M frozen | ~0.78–0.84* |
| NT-v2-500M frozen | ~0.82–0.87* |

*Estimates from NT transfer-learning benchmarks; full evaluation in progress.*

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

## Why Δ = embed(alt) − embed(ref)?

Standard tools (CADD, REVEL) rely on hand-crafted annotations. GenGI instead uses a **pre-trained DNA LLM as a universal sequence encoder** and computes the mutation-effect vector Δ: a 512-dimensional representation of *what changes* when a substitution occurs in the DNA context. This is:

- **Context-aware** — the NT has been pre-trained on 2,500+ genomes and encodes splicing signals, regulatory grammar, and evolutionary conservation
- **Annotation-free** — no external databases required at inference time
- **Naturally patient-scalable** — the TransformerEncoder aggregates Δ vectors across all variants in a patient's exome via a learned CLS token

## Stack

| Component | Technology |
|-----------|------------|
| DNA-LLM backbone | Nucleotide Transformer v2 50M (InstaDeepAI / EMBL-EBI) |
| Deep learning | PyTorch 2.1+ |
| HuggingFace | `transformers`, `accelerate` |
| Training data | ClinVar GRCh38 SNVs |
| Sequences | Ensembl REST API |
| Evaluation | scikit-learn (AUROC) |
| Demo | Streamlit + Plotly |
| Target (Phase 2) | UK Biobank WES (200 k+ participants) |

## Roadmap

| Phase | Status | Goal |
|-------|--------|------|
| **Phase 1** | ✅ Current | ClinVar SNV pathogenicity — binary classification |
| **Phase 2** | Planned | UK Biobank WES — multi-variant patient scoring, phenotype association |
| **Phase 3** | Planned | Multi-phenotype structured output (disease ontology trees) |
| **Phase 4** | Planned | Counterfactual XAI, TCAV genomic concepts, ACMG alignment |

## Citation

```bibtex
@software{tall2026gengi,
  author = {Tall, Mamadou Lamine},
  title  = {{GenGI}: General Genome Interpretation via DNA-LLM + TransformerEncoder},
  year   = {2026},
  url    = {https://github.com/mamadoulaminetall/GenGI},
}
```

```bibtex
@article{dalla2023nucleotide,
  title   = {The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author  = {Dalla-Torre, Hugo and others},
  journal = {bioRxiv},
  year    = {2023},
  doi     = {10.1101/2023.01.11.523679}
}
```

## Author

**Dr. Mamadou Lamine TALL** — PhD Bioinformatics  
Founder, [MedFlow AI](https://github.com/mamadoulaminetall)  
Contact: [laminetall30@gmail.com](mailto:laminetall30@gmail.com)

*Inspired by the [AI4GI group](https://www.igmm.cnrs.fr), IGMM CNRS Montpellier — Dr. Daniele Raimondi.*

## License

MIT
