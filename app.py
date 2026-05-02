"""
GenGI — Streamlit Demo
General Genome Interpretation: variant pathogenicity scoring powered by
Nucleotide Transformer + PyTorch TransformerEncoder.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import torch

from model import GenGI

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GenGI — General Genome Interpretation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Hard-coded demo variants (ClinVar-sourced)
# ---------------------------------------------------------------------------
DEMO_VARIANTS = [
    {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A", "gene": "BRCA1", "label": "Pathogenic"},
    {"chrom": "13", "pos": 32340300, "ref": "A", "alt": "T", "gene": "BRCA2", "label": "Pathogenic"},
    {"chrom": "17", "pos": 7675088,  "ref": "C", "alt": "T", "gene": "TP53",  "label": "Pathogenic"},
    {"chrom": "17", "pos": 43115726, "ref": "T", "alt": "C", "gene": "BRCA1", "label": "Benign"},
    {"chrom": "13", "pos": 32370402, "ref": "C", "alt": "T", "gene": "BRCA2", "label": "Benign"},
]

DEMO_TEXT = "\n".join(
    f"chr{v['chrom']}:{v['pos']}:{v['ref']}>{v['alt']}" for v in DEMO_VARIANTS
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e40af, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-top: 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pathogenic-badge {
        background: #fee2e2;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .benign-badge {
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model + Embedder loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model(hidden_dim: int = 256) -> GenGI:
    model = GenGI(input_dim=512, hidden_dim=hidden_dim)
    # Load checkpoint if available
    ckpt_path = Path("models/gengi_best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    else:
        model.eval()
    return model


@st.cache_resource
def load_embedder(mode: str, device: str = "cpu"):
    if mode == "demo":
        from embedder import MockEmbedder
        return MockEmbedder()
    else:
        from embedder import DNAEmbedder
        return DNAEmbedder(device=device)


# ---------------------------------------------------------------------------
# Utility: parse variant strings
# ---------------------------------------------------------------------------


def parse_variant_lines(text: str) -> List[dict]:
    """
    Parse lines of format  chrN:POS:REF>ALT  or  N:POS:REF>ALT.
    Returns list of dicts with keys: chrom, pos, ref, alt, variant_id.
    """
    variants = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            # Support both chr17:43094692:G>A and 17:43094692:G>A
            line = line.replace("chr", "")
            parts = line.split(":")
            chrom = parts[0]
            pos = int(parts[1])
            alleles = parts[2].split(">")
            ref, alt = alleles[0].upper(), alleles[1].upper()
            variants.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "variant_id": f"chr{chrom}:{pos}:{ref}>{alt}",
                    "gene": "Unknown",
                }
            )
        except Exception:
            continue
    return variants


def parse_vcf(uploaded_file) -> List[dict]:
    """Parse minimal VCF: extract CHROM, POS, REF, ALT."""
    content = uploaded_file.read().decode("utf-8", errors="replace")
    variants = []
    for line in content.splitlines():
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        try:
            chrom = cols[0].replace("chr", "")
            pos = int(cols[1])
            ref = cols[3].strip().upper()
            alt = cols[4].strip().split(",")[0].upper()
            if len(ref) == 1 and len(alt) == 1:  # SNVs only
                variants.append(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "variant_id": f"chr{chrom}:{pos}:{ref}>{alt}",
                        "gene": "Unknown",
                    }
                )
        except Exception:
            continue
    return variants


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


def score_variants(
    variants: List[dict],
    model: GenGI,
    embedder,
    demo_mode: bool,
) -> pd.DataFrame:
    """
    Run the full GenGI pipeline on a list of variant dicts.
    In demo mode skips Ensembl API calls and uses mock sequences.
    """
    if not variants:
        return pd.DataFrame()

    ref_seqs, alt_seqs = [], []

    if demo_mode:
        # Generate plausible-length mock sequences (257 bp context)
        rng = np.random.RandomState(seed=sum(v["pos"] for v in variants) % (2**31))
        nucleotides = ["A", "C", "G", "T"]
        for v in variants:
            seq = "".join(rng.choice(nucleotides, 257))
            ref_seq = seq[:128] + v["ref"] + seq[129:]
            alt_seq = seq[:128] + v["alt"] + seq[129:]
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)
    else:
        from data_loader import get_variant_context
        for v in variants:
            ref_ctx, alt_ctx = get_variant_context(
                chrom=v["chrom"],
                pos=v["pos"],
                ref=v["ref"],
                alt=v["alt"],
                window=128,
            )
            ref_seqs.append(ref_ctx if ref_ctx else "A" * 257)
            alt_seqs.append(alt_ctx if alt_ctx else "A" * 257)

    # Embed
    delta = embedder.embed_variant_delta(ref_seqs, alt_seqs)  # [N, 512]

    # GenGI forward (each variant as a patient with N=1)
    scores = []
    importances = []
    model.eval()
    for i in range(len(variants)):
        d = delta[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
        with torch.no_grad():
            logit = model(d)
            prob = torch.sigmoid(logit).item()
        scores.append(prob)

        # XAI: gradient × input (enable grad for this call)
        d_grad = delta[i].unsqueeze(0).unsqueeze(0).detach().requires_grad_(True)
        logit_xai = model(d_grad)
        score_xai = torch.sigmoid(logit_xai).squeeze()
        score_xai.backward()
        importance = (d_grad.grad * d_grad).abs().mean().item()
        importances.append(importance)

    # Build result DataFrame
    rows = []
    for v, score, imp in zip(variants, scores, importances):
        classification = "Pathogenic" if score >= 0.5 else "Benign"
        confidence = abs(score - 0.5) * 2  # 0→1 scale
        rows.append(
            {
                "Gene": v.get("gene", "Unknown"),
                "Variant": v["variant_id"],
                "Score": round(score, 4),
                "Classification": classification,
                "Confidence": round(confidence, 3),
                "XAI_importance": round(imp, 6),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def make_score_bar_chart(results: pd.DataFrame) -> go.Figure:
    colors = [
        "#ef4444" if c == "Pathogenic" else "#3b82f6"
        for c in results["Classification"]
    ]
    fig = go.Figure(
        go.Bar(
            x=results["Variant"],
            y=results["Score"],
            marker_color=colors,
            text=[f"{s:.3f}" for s in results["Score"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Score: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Decision threshold (0.5)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-Variant Pathogenicity Score",
        xaxis_title="Variant",
        yaxis_title="Pathogenicity Score",
        yaxis=dict(range=[0, 1.15]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=60, b=100),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def make_gauge(score: float) -> go.Figure:
    if score < 0.33:
        color = "#22c55e"
        label = "Low Risk"
    elif score < 0.66:
        color = "#f59e0b"
        label = "Intermediate Risk"
    else:
        color = "#ef4444"
        label = "High Risk"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(score * 100, 1),
            delta={"reference": 50, "valueformat": ".1f"},
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": f"Genomic Risk Score<br><span style='font-size:0.9em;color:{color}'>{label}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 33], "color": "#dcfce7"},
                    {"range": [33, 66], "color": "#fef9c3"},
                    {"range": [66, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def make_xai_chart(results: pd.DataFrame) -> go.Figure:
    df_sorted = results.sort_values("XAI_importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df_sorted["XAI_importance"],
            y=df_sorted["Variant"],
            orientation="h",
            marker_color=[
                "#ef4444" if c == "Pathogenic" else "#3b82f6"
                for c in df_sorted["Classification"]
            ],
            text=[f"{v:.4f}" for v in df_sorted["XAI_importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Variant Contribution (Gradient × Input Attribution)",
        xaxis_title="Attribution magnitude",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=200, t=60, b=40, r=80),
        height=max(300, len(df_sorted) * 55),
    )
    return fig


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------

ARCH_DIAGRAM = """\
┌─────────────────────────────────────────────────────────────────┐
│                    GenGI Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: WES variants  chr:pos:ref:alt                          │
│           │                                                     │
│           ▼                                                     │
│  Ensembl REST API  →  ±128 bp DNA context                      │
│           │                   │                                 │
│       ref_seq             alt_seq                               │
│           │                   │                                 │
│           ▼                   ▼                                 │
│  ┌─────────────────────────────────┐                           │
│  │  Nucleotide Transformer v2      │  InstaDeepAI (50M params) │
│  │  6-mer tokenisation             │                           │
│  │  CLS embedding  [512-dim]       │                           │
│  └─────────────────────────────────┘                           │
│           │                   │                                 │
│      embed(ref)          embed(alt)                             │
│           └────────┬──────────┘                                 │
│                    ▼                                            │
│          Δ = embed(alt) − embed(ref)   [512-dim]               │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────┐                           │
│  │  GenGI TransformerEncoder       │                           │
│  │  Linear(512→256)                │                           │
│  │  [CLS] + variant tokens         │                           │
│  │  TransformerEncoder (2L, 4H)    │                           │
│  │  Pre-LN, GELU, d_ff=1024        │                           │
│  └─────────────────────────────────┘                           │
│                    │                                            │
│                    ▼                                            │
│  LayerNorm → Linear → GELU → Dropout → Linear(1)               │
│                    │                                            │
│                    ▼                                            │
│          Pathogenicity score  ∈ [0, 1]                         │
│                                                                 │
│  XAI: gradient × Δ attribution  →  per-variant importance      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
                <span style='font-size:2.5rem; font-weight:900;
                    background:linear-gradient(90deg,#1e40af,#0ea5e9);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                    🧬 GenGI
                </span><br>
                <span style='font-size:0.8rem; color:#64748b; font-weight:500;
                    letter-spacing:0.08em; text-transform:uppercase;'>
                    General Genome Interpretation
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        mode_label = st.radio(
            "Inference mode",
            options=["🔬 Demo (mock embeddings)", "🚀 Full model (Nucleotide Transformer)"],
            index=0,
            help="Demo mode uses random embeddings — no model download required.",
        )
        demo_mode = mode_label.startswith("🔬")

        device = "cpu"
        if not demo_mode:
            device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)

        st.divider()
        st.markdown(
            """
            **About GenGI**

            GenGI combines [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
            (InstaDeepAI / EMBL-EBI) with a custom PyTorch TransformerEncoder to predict
            variant pathogenicity from raw DNA context sequences.

            Each variant is encoded as a *mutation-effect vector*
            Δ = embed(alt) − embed(ref), capturing the functional shift
            induced by the substitution in the NT embedding space.

            ---
            🔗 [AI4GI Group — IGMM CNRS Montpellier](https://www.igmm.cnrs.fr)
            """,
        )

    return mode_label, device


# ---------------------------------------------------------------------------
# Tab 1: Variant Analysis
# ---------------------------------------------------------------------------


def tab_variant_analysis(demo_mode: bool, model: GenGI, embedder):
    st.subheader("Variant Input")

    input_mode = st.radio(
        "Input method",
        ["Paste variants", "Upload VCF", "Try example"],
        horizontal=True,
    )

    variants: List[dict] = []

    if input_mode == "Paste variants":
        st.markdown(
            "Enter one variant per line, format: `chrN:POS:REF>ALT`  "
            "(e.g. `chr17:43094692:G>A`)"
        )
        text = st.text_area("Variants", height=160, placeholder="chr17:43094692:G>A\nchr13:32340300:A>T")
        if text.strip():
            variants = parse_variant_lines(text)
            if variants:
                st.info(f"Parsed **{len(variants)}** variant(s).")
            else:
                st.warning("Could not parse any variants. Check format: `chrN:POS:REF>ALT`")

    elif input_mode == "Upload VCF":
        vcf_file = st.file_uploader("Upload VCF file", type=["vcf", "txt"])
        if vcf_file:
            variants = parse_vcf(vcf_file)
            if variants:
                st.info(f"Parsed **{len(variants)}** SNV(s) from VCF.")
            else:
                st.warning("No SNVs found in VCF (only single-nucleotide substitutions supported).")

    else:  # Try example
        st.markdown("**5 ClinVar variants** (BRCA1, BRCA2, TP53) — 3 pathogenic + 2 benign:")
        example_df = pd.DataFrame(DEMO_VARIANTS)[["gene", "chrom", "pos", "ref", "alt", "label"]]
        example_df.columns = ["Gene", "Chr", "Pos", "Ref", "Alt", "Expected"]
        st.dataframe(example_df, use_container_width=True, hide_index=True)

        for v in DEMO_VARIANTS:
            v["variant_id"] = f"chr{v['chrom']}:{v['pos']}:{v['ref']}>{v['alt']}"
        variants = list(DEMO_VARIANTS)

    st.divider()

    if st.button("🔍 Analyze Variants", type="primary", disabled=len(variants) == 0):
        if len(variants) == 0:
            st.error("No variants to analyze.")
            return

        with st.spinner("Running GenGI pipeline ..."):
            try:
                results = score_variants(variants, model, embedder, demo_mode)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        if results.empty:
            st.warning("No results generated.")
            return

        # --- Summary metrics ---
        n_total = len(results)
        n_path = (results["Classification"] == "Pathogenic").sum()
        mean_score = results["Score"].mean()
        top_gene = results.sort_values("Score", ascending=False).iloc[0]["Gene"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{n_total}</div>"
                f"<div class='metric-label'>Variants Analyzed</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{n_path}</div>"
                f"<div class='metric-label'>Predicted Pathogenic</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{mean_score:.3f}</div>"
                f"<div class='metric-label'>Mean Score</div></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{top_gene}</div>"
                f"<div class='metric-label'>Top Gene</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bar chart ---
        st.plotly_chart(make_score_bar_chart(results), use_container_width=True)

        # --- Results table ---
        st.subheader("Results")
        display_df = results[["Gene", "Variant", "Score", "Classification", "Confidence"]].copy()

        def fmt_classification(val):
            if val == "Pathogenic":
                return "🔴 Pathogenic"
            return "🔵 Benign"

        display_df["Classification"] = display_df["Classification"].apply(fmt_classification)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.4f"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )

        # --- Download ---
        csv_bytes = results.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_bytes,
            file_name="gengi_results.csv",
            mime="text/csv",
        )

        # Store in session state for Patient View tab
        st.session_state["results"] = results


# ---------------------------------------------------------------------------
# Tab 2: Patient View
# ---------------------------------------------------------------------------


def tab_patient_view():
    st.subheader("Patient-Level Genomic Risk Assessment")

    results: Optional[pd.DataFrame] = st.session_state.get("results", None)

    if results is None or results.empty:
        st.info("Run a **Variant Analysis** first (Tab 1) to see the patient view.")
        return

    # Aggregate score: mean of per-variant pathogenicity scores
    aggregate_score = float(results["Score"].mean())
    n_path = int((results["Classification"] == "Pathogenic").sum())
    n_total = len(results)

    col_gauge, col_summary = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(make_gauge(aggregate_score), use_container_width=True)

    with col_summary:
        st.markdown("### Clinical Interpretation")

        top_variants = (
            results.sort_values("Score", ascending=False)
            .head(3)["Variant"]
            .tolist()
        )
        top_str = ", ".join(top_variants)

        risk_word = (
            "high" if aggregate_score >= 0.66
            else "intermediate" if aggregate_score >= 0.33
            else "low"
        )

        st.markdown(
            f"""
            > **{n_path}/{n_total}** variants predicted pathogenic.
            > Overall genomic risk score: **{aggregate_score:.3f}** ({risk_word} risk).
            >
            > Top contributing variants: **{top_str}**.
            >
            > *Note: GenGI is a research prototype. Results should be interpreted
            > alongside clinical data and validated variant databases (ClinVar, HGMD).*
            """
        )

        if aggregate_score >= 0.66:
            st.error("⚠️  High-risk profile — consider genetic counselling referral.")
        elif aggregate_score >= 0.33:
            st.warning("Intermediate-risk profile — additional variant interpretation recommended.")
        else:
            st.success("Low-risk genomic profile based on analyzed variants.")

    # XAI attribution chart
    if len(results) > 1:
        st.markdown("---")
        st.subheader("Variant Contribution (XAI Attribution)")
        st.caption(
            "Gradient × input magnitude: how much each variant's mutation-effect vector "
            "influenced the pathogenicity prediction."
        )
        st.plotly_chart(make_xai_chart(results), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Architecture
# ---------------------------------------------------------------------------


def tab_architecture(demo_mode: bool):
    st.subheader("GenGI Model Architecture")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.code(ARCH_DIAGRAM, language=None)

    with col2:
        st.markdown("### Key Parameters")
        model = load_model()
        params = model.count_parameters()

        st.markdown(
            f"""
            | Component | Value |
            |-----------|-------|
            | NT model | NT-v2-50M |
            | Embedding dim (D) | 512 |
            | GenGI hidden dim | 256 |
            | Transformer layers | 2 |
            | Attention heads | 4 |
            | Feed-forward dim | 1024 |
            | GenGI parameters | {params:,} |
            | Context window | ±128 bp |
            | Input | Δ = embed(alt)−embed(ref) |
            | Output | Pathogenicity ∈ [0, 1] |
            """
        )

        st.markdown("### References")
        st.markdown(
            """
            - Dalla-Torre et al. *Nucleotide Transformer* (2023) · [bioRxiv](https://doi.org/10.1101/2023.01.11.523679)
            - Zhou et al. *DNABERT-2* (2023) · [arXiv:2306.15006](https://arxiv.org/abs/2306.15006)
            - Landrum et al. *ClinVar* · [NAR 2018](https://doi.org/10.1093/nar/gkx1153)
            - UK Biobank WES · [Nature 2022](https://doi.org/10.1038/s41586-022-04965-x)
            """
        )

        st.markdown("### Repository")
        st.markdown(
            """
            ```
            git clone https://github.com/mlaminetall/GenGI
            cd GenGI
            pip install -r requirements.txt
            streamlit run app.py
            ```
            """
        )

        ckpt_path = Path("models/gengi_best.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            val_auroc = ckpt.get("val_auroc", "N/A")
            epoch = ckpt.get("epoch", "N/A")
            st.success(f"Checkpoint loaded: epoch {epoch}, val AUROC = {val_auroc:.4f}")
        else:
            st.info("No checkpoint found — run `python train.py --mock-embedder` to train.")


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------


def main():
    # Sidebar
    mode_label, device = render_sidebar()
    demo_mode = mode_label.startswith("🔬")

    # Header
    st.markdown(
        "<h1 class='main-header'>🧬 GenGI</h1>"
        "<p class='sub-header'>General Genome Interpretation — "
        "Nucleotide Transformer + TransformerEncoder for variant pathogenicity</p>",
        unsafe_allow_html=True,
    )

    # Demo mode banner
    if demo_mode:
        st.warning(
            "⚠️  **Demo mode** — embeddings are random (MockEmbedder). "
            "Load a trained checkpoint for real predictions.",
            icon="⚠️",
        )

    # Load model + embedder
    with st.spinner("Loading GenGI model ..."):
        model = load_model()

    mode_key = "demo" if demo_mode else "full"
    with st.spinner("Initialising embedder ..."):
        try:
            embedder = load_embedder(mode_key, device=device)
        except Exception as e:
            st.error(f"Failed to load embedder: {e}")
            st.stop()

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔬 Variant Analysis", "🧑‍⚕️ Patient View", "📐 Architecture"]
    )

    with tab1:
        tab_variant_analysis(demo_mode, model, embedder)

    with tab2:
        tab_patient_view()

    with tab3:
        tab_architecture(demo_mode)

    # Footer
    st.markdown(
        """
        <hr style='margin-top:3rem; border:none; border-top:1px solid #e2e8f0;'>
        <p style='text-align:center; font-size:0.78rem; color:#94a3b8; padding:0.5rem 0 1rem 0;'>
            Built by <strong>Dr. Mamadou Lamine TALL</strong> · MedFlow AI ·
            Powered by Nucleotide Transformer + PyTorch ·
            Contact: <a href='mailto:laminetall30@gmail.com'>laminetall30@gmail.com</a>
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
