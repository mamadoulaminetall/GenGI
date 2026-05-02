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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GenGI — General Genome Interpretation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Demo variants (ClinVar-sourced)
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
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Navbar ── */
.gengi-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1.4rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #1e3a5f 100%);
    border-radius: 14px;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 24px rgba(30,64,175,0.18);
}
.navbar-left { display: flex; align-items: center; gap: 1rem; }
.navbar-title {
    font-size: 1.65rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.navbar-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: -2px;
    letter-spacing: 0.03em;
}
.navbar-badge {
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(30,64,175,0.10); }
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #1e40af, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
    margin-top: 2px;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.2rem 0 0.6rem 0;
    border-left: 4px solid #3b82f6;
    padding-left: 0.7rem;
}

/* ── Demo banner ── */
.demo-banner {
    background: linear-gradient(90deg, #fefce8, #fffbeb);
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.87rem;
    color: #92400e;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.gengi-footer {
    margin-top: 3rem;
    padding: 1.2rem 0 0.8rem 0;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    font-size: 0.76rem;
    color: #94a3b8;
}
.gengi-footer a { color: #60a5fa; text-decoration: none; }

/* Hide default footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# MedFlow AI SVG logo (unique gradient IDs: mfhg / mftg to avoid collisions)
MEDFLOW_LOGO_SVG = """
<svg width="200" height="66" viewBox="0 0 220 72" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="mfhg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#7c3aed"/>
      <stop offset="100%" style="stop-color:#2563eb"/>
    </linearGradient>
    <linearGradient id="mftg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#a78bfa"/>
      <stop offset="100%" style="stop-color:#60a5fa"/>
    </linearGradient>
  </defs>
  <path d="M30 18 C30 12 24 8 18 12 C12 16 12 24 18 30 L30 42 L42 30 C48 24 48 16 42 12 C36 8 30 12 30 18Z"
        fill="url(#mfhg)" opacity="0.95"/>
  <polyline points="14,30 19,30 22,22 25,38 28,26 31,30 38,30 41,30"
            fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
  <text x="56" y="32" font-family="'Helvetica Neue',Arial,sans-serif" font-size="22" font-weight="800"
        fill="url(#mftg)" letter-spacing="-0.5">MedFlow</text>
  <rect x="158" y="16" width="26" height="18" rx="5" fill="url(#mfhg)"/>
  <text x="171" y="29" font-family="'Helvetica Neue',Arial,sans-serif" font-size="11" font-weight="700"
        fill="white" text-anchor="middle" letter-spacing="0.5">AI</text>
  <text x="56" y="52" font-family="'Helvetica Neue',Arial,sans-serif" font-size="11" font-weight="400"
        fill="#94a3b8" letter-spacing="0.3">GenGI · Genome Interpretation</text>
</svg>
"""

# ---------------------------------------------------------------------------
# Model + Embedder loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(hidden_dim: int = 256) -> GenGI:
    model = GenGI(input_dim=512, hidden_dim=hidden_dim)
    ckpt_path = Path("models/gengi_best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_embedder(mode: str, device: str = "cpu"):
    if mode == "demo":
        from embedder import MockEmbedder
        return MockEmbedder(), False
    try:
        from embedder import DNAEmbedder
        return DNAEmbedder(device=device), False
    except Exception:
        from embedder import MockEmbedder
        return MockEmbedder(), True   # True = fell back to mock


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_variant_lines(text: str) -> List[dict]:
    variants = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            line = line.replace("chr", "")
            parts = line.split(":")
            chrom = parts[0]
            pos = int(parts[1])
            ref, alt = parts[2].split(">")
            ref, alt = ref.upper(), alt.upper()
            variants.append({
                "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                "variant_id": f"chr{chrom}:{pos}:{ref}>{alt}",
                "gene": "Unknown",
            })
        except Exception:
            continue
    return variants


def parse_vcf(uploaded_file) -> List[dict]:
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
            if len(ref) == 1 and len(alt) == 1:
                variants.append({
                    "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                    "variant_id": f"chr{chrom}:{pos}:{ref}>{alt}",
                    "gene": "Unknown",
                })
        except Exception:
            continue
    return variants


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def score_variants(variants: List[dict], model: GenGI, embedder, demo_mode: bool) -> pd.DataFrame:
    if not variants:
        return pd.DataFrame()

    ref_seqs, alt_seqs = [], []

    if demo_mode:
        rng = np.random.RandomState(seed=sum(v["pos"] for v in variants) % (2**31))
        bases = ["A", "C", "G", "T"]
        for v in variants:
            seq = "".join(rng.choice(bases, 257))
            ref_seqs.append(seq[:128] + v["ref"] + seq[129:])
            alt_seqs.append(seq[:128] + v["alt"] + seq[129:])
    else:
        from data_loader import get_variant_context
        for v in variants:
            rc, ac = get_variant_context(v["chrom"], v["pos"], v["ref"], v["alt"], window=128)
            ref_seqs.append(rc if rc else "A" * 257)
            alt_seqs.append(ac if ac else "A" * 257)

    delta = embedder.embed_variant_delta(ref_seqs, alt_seqs)

    scores, importances = [], []
    model.eval()
    for i in range(len(variants)):
        d = delta[i].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(d)).item()
        scores.append(prob)

        d_g = delta[i].unsqueeze(0).unsqueeze(0).detach().requires_grad_(True)
        torch.sigmoid(model(d_g)).squeeze().backward()
        importances.append((d_g.grad * d_g).abs().mean().item())

    rows = []
    for v, score, imp in zip(variants, scores, importances):
        rows.append({
            "Gene": v.get("gene", "—"),
            "Variant": v["variant_id"],
            "Score": round(score, 4),
            "Classification": "Pathogenic" if score >= 0.5 else "Benign",
            "Confidence": round(abs(score - 0.5) * 2, 3),
            "XAI_importance": round(imp, 6),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def make_score_bar(results: pd.DataFrame) -> go.Figure:
    colors = ["#ef4444" if c == "Pathogenic" else "#3b82f6" for c in results["Classification"]]
    fig = go.Figure(go.Bar(
        x=results["Variant"], y=results["Score"],
        marker_color=colors,
        text=[f"{s:.3f}" for s in results["Score"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Threshold 0.5", annotation_position="top right")
    fig.update_layout(
        title="Per-Variant Pathogenicity Score",
        xaxis_title="Variant", yaxis_title="Score",
        yaxis=dict(range=[0, 1.18]),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=56, b=110), showlegend=False,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def make_gauge(score: float) -> go.Figure:
    if score < 0.33:
        color, label = "#22c55e", "Low Risk"
    elif score < 0.66:
        color, label = "#f59e0b", "Intermediate Risk"
    else:
        color, label = "#ef4444", "High Risk"

    fig = go.Figure(go.Indicator(
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
            "threshold": {"line": {"color": "#1e293b", "width": 3}, "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(height=320, paper_bgcolor="white",
                      font=dict(family="Inter, sans-serif"),
                      margin=dict(t=40, b=20, l=20, r=20))
    return fig


def make_xai_chart(results: pd.DataFrame) -> go.Figure:
    df = results.sort_values("XAI_importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["XAI_importance"], y=df["Variant"], orientation="h",
        marker_color=["#ef4444" if c == "Pathogenic" else "#3b82f6" for c in df["Classification"]],
        text=[f"{v:.4f}" for v in df["XAI_importance"]], textposition="outside",
    ))
    fig.update_layout(
        title="Variant Contribution — Gradient × Input Attribution",
        xaxis_title="Attribution magnitude",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=200, t=56, b=40, r=80),
        height=max(300, len(df) * 55),
    )
    return fig


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------

ARCH_DIAGRAM = """\
┌──────────────────────────────────────────────────────────────────┐
│                    GenGI Architecture                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: WES variants   chrN:pos:REF>ALT                         │
│              │                                                   │
│              ▼                                                   │
│  Ensembl REST API  →  ±128 bp DNA context                        │
│              │              │                                    │
│          ref_seq         alt_seq                                 │
│              │              │                                    │
│              ▼              ▼                                    │
│  ┌───────────────────────────────────┐                          │
│  │  Nucleotide Transformer v2 (50M)  │  InstaDeepAI / EMBL-EBI  │
│  │  6-mer tokenisation               │                          │
│  │  CLS embedding  → [512-dim]       │                          │
│  └───────────────────────────────────┘                          │
│              │              │                                    │
│         embed(ref)      embed(alt)                               │
│              └──────┬───────┘                                    │
│                     ▼                                            │
│           Δ = embed(alt) − embed(ref)   [512-dim]               │
│                     │                                            │
│                     ▼                                            │
│  ┌───────────────────────────────────┐                          │
│  │  GenGI TransformerEncoder         │                          │
│  │  Linear(512 → 256)                │                          │
│  │  [CLS_token] ++ variant_tokens    │                          │
│  │  Pre-LN TransformerEncoder 2L 4H  │  d_ff = 1024            │
│  └───────────────────────────────────┘                          │
│                     │                                            │
│                     ▼                                            │
│   LayerNorm → Linear → GELU → Dropout → Linear(1) → σ          │
│                     │                                            │
│                     ▼                                            │
│           Pathogenicity ∈ [0, 1]                                │
│                                                                  │
│   XAI:  gradient × Δ attribution  →  per-variant importance     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        # MedFlow AI logo
        st.markdown(
            "<div style='text-align:center; padding:1rem 0 0.5rem 0;'>"
            + MEDFLOW_LOGO_SVG.strip()
            + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        mode_label = st.radio(
            "Inference mode",
            options=["🔬 Demo (mock embeddings)", "🚀 Full model (Nucleotide Transformer)"],
            index=0,
            help="Demo mode uses deterministic random embeddings — no model download required.",
        )
        demo_mode = mode_label.startswith("🔬")

        device = "cpu"
        if not demo_mode:
            device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)

        st.divider()
        st.markdown("""
**About GenGI**

GenGI combines [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
(InstaDeepAI / EMBL-EBI) with a custom PyTorch TransformerEncoder to score
variant pathogenicity from raw DNA context.

Each variant is encoded as a *mutation-effect vector*
**Δ = embed(alt) − embed(ref)**, capturing the functional shift
induced by the substitution in NT embedding space.

---
🔗 [AI4GI · IGMM CNRS Montpellier](https://www.igmm.cnrs.fr)
🔗 [GitHub · GenGI](https://github.com/mamadoulaminetall/GenGI)
        """)

    return mode_label, device


# ---------------------------------------------------------------------------
# Tab 1 — Variant Analysis
# ---------------------------------------------------------------------------

def tab_variant_analysis(demo_mode: bool, model: GenGI, embedder):
    st.markdown("<div class='section-title'>Variant Input</div>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["Paste variants", "Upload VCF", "Try example"],
        horizontal=True,
        label_visibility="collapsed",
    )

    variants: List[dict] = []

    if input_mode == "Paste variants":
        st.caption("One variant per line — format: `chrN:POS:REF>ALT`  (e.g. `chr17:43094692:G>A`)")
        text = st.text_area("Variants", height=150,
                            placeholder="chr17:43094692:G>A\nchr13:32340300:A>T\nchr17:7675088:C>T")
        if text.strip():
            variants = parse_variant_lines(text)
            if variants:
                st.info(f"Parsed **{len(variants)}** variant(s).")
            else:
                st.warning("Could not parse any variants. Expected format: `chrN:POS:REF>ALT`")

    elif input_mode == "Upload VCF":
        vcf_file = st.file_uploader("Upload VCF file", type=["vcf", "txt"])
        if vcf_file:
            variants = parse_vcf(vcf_file)
            st.info(f"Parsed **{len(variants)}** SNV(s) from VCF." if variants
                    else "No SNVs found (only single-nucleotide substitutions supported).")

    else:
        st.caption("5 ClinVar-sourced variants — BRCA1 / BRCA2 / TP53 — 3 pathogenic + 2 benign")
        ex = pd.DataFrame(DEMO_VARIANTS)[["gene", "chrom", "pos", "ref", "alt", "label"]]
        ex.columns = ["Gene", "Chr", "Pos", "Ref", "Alt", "Expected"]
        st.dataframe(ex, use_container_width=True, hide_index=True)
        for v in DEMO_VARIANTS:
            v["variant_id"] = f"chr{v['chrom']}:{v['pos']}:{v['ref']}>{v['alt']}"
        variants = list(DEMO_VARIANTS)

    st.divider()

    if st.button("🔍 Analyze Variants", type="primary", disabled=len(variants) == 0):
        with st.spinner("Running GenGI pipeline …"):
            try:
                results = score_variants(variants, model, embedder, demo_mode)
            except Exception as e:
                st.error(f"Analysis error: {e}")
                return

        if results.empty:
            st.warning("No results.")
            return

        n_total   = len(results)
        n_path    = (results["Classification"] == "Pathogenic").sum()
        mean_sc   = results["Score"].mean()
        top_gene  = results.sort_values("Score", ascending=False).iloc[0]["Gene"]

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, n_total,        "Variants analyzed"),
            (c2, n_path,         "Predicted pathogenic"),
            (c3, f"{mean_sc:.3f}", "Mean score"),
            (c4, top_gene,       "Top gene"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{val}</div>"
                    f"<div class='metric-label'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(make_score_bar(results), use_container_width=True)

        st.markdown("<div class='section-title'>Results Table</div>", unsafe_allow_html=True)
        display = results[["Gene", "Variant", "Score", "Classification", "Confidence"]].copy()
        display["Classification"] = display["Classification"].map(
            {"Pathogenic": "🔴 Pathogenic", "Benign": "🔵 Benign"}
        )
        st.dataframe(
            display, use_container_width=True, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.3f"),
            },
        )

        csv = results.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv, file_name="gengi_results.csv", mime="text/csv")
        st.session_state["results"] = results


# ---------------------------------------------------------------------------
# Tab 2 — Patient View
# ---------------------------------------------------------------------------

def tab_patient_view():
    st.markdown("<div class='section-title'>Patient-Level Genomic Risk</div>", unsafe_allow_html=True)

    results: Optional[pd.DataFrame] = st.session_state.get("results")
    if results is None or results.empty:
        st.info("Run a **Variant Analysis** (Tab 1) first to populate this view.")
        return

    agg     = float(results["Score"].mean())
    n_path  = int((results["Classification"] == "Pathogenic").sum())
    n_total = len(results)

    col_g, col_s = st.columns([1, 1])
    with col_g:
        st.plotly_chart(make_gauge(agg), use_container_width=True)

    with col_s:
        st.markdown("### Clinical Interpretation")
        top3 = results.sort_values("Score", ascending=False).head(3)["Variant"].tolist()
        risk = "high" if agg >= 0.66 else "intermediate" if agg >= 0.33 else "low"
        st.markdown(f"""
> **{n_path}/{n_total}** variants predicted pathogenic.
> Overall genomic risk: **{agg:.3f}** ({risk} risk).
>
> Top contributors: **{', '.join(top3)}**.
>
> *Research prototype — validate results against ClinVar / HGMD.*
        """)
        if agg >= 0.66:
            st.error("⚠️ High-risk profile — consider genetic counselling.")
        elif agg >= 0.33:
            st.warning("Intermediate-risk — additional interpretation recommended.")
        else:
            st.success("Low-risk genomic profile.")

    if len(results) > 1:
        st.divider()
        st.markdown("<div class='section-title'>XAI Attribution — Variant Contributions</div>",
                    unsafe_allow_html=True)
        st.caption("Gradient × input magnitude: how strongly each variant's Δ-embedding drives the prediction.")
        st.plotly_chart(make_xai_chart(results), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Architecture
# ---------------------------------------------------------------------------

def tab_architecture(demo_mode: bool):
    st.markdown("<div class='section-title'>GenGI Model Architecture</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.code(ARCH_DIAGRAM, language=None)

    with col2:
        model = load_model()
        params = model.count_parameters()
        st.markdown(f"""
| Component | Value |
|-----------|-------|
| NT backbone | NT-v2-50M (InstaDeepAI) |
| NT embedding dim | 512 |
| Δ = embed(alt)−embed(ref) | 512-dim |
| GenGI hidden dim | 256 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Feed-forward dim | 1 024 |
| Trainable params | {params:,} |
| DNA context window | ±128 bp |
| XAI method | Gradient × input |
        """)

        st.markdown("### Key References")
        st.markdown("""
- Dalla-Torre et al. *Nucleotide Transformer* · [bioRxiv 2023](https://doi.org/10.1101/2023.01.11.523679)
- Zhou et al. *DNABERT-2* · [arXiv 2023](https://arxiv.org/abs/2306.15006)
- Landrum et al. *ClinVar* · [NAR 2018](https://doi.org/10.1093/nar/gkx1153)
- UK Biobank WES · [Nature 2022](https://doi.org/10.1038/s41586-022-04965-x)
        """)

        ckpt = Path("models/gengi_best.pt")
        if ckpt.exists():
            c = torch.load(ckpt, map_location="cpu")
            st.success(f"Checkpoint: epoch {c.get('epoch','?')} — val AUROC {c.get('val_auroc',0):.4f}")
        else:
            st.caption("No checkpoint found — run `python train.py --mock-embedder` to train a model.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode_label, device = render_sidebar()
    demo_mode = mode_label.startswith("🔬")

    # Navbar
    st.markdown(f"""
<div class="gengi-navbar">
  <div class="navbar-left">
    <svg width="38" height="38" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="nbhg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#7c3aed"/>
          <stop offset="100%" style="stop-color:#2563eb"/>
        </linearGradient>
      </defs>
      <path d="M25 10 C25 5 20 2 15 6 C10 10 10 18 15 23 L25 33 L35 23 C40 18 40 10 35 6 C30 2 25 5 25 10Z"
            fill="url(#nbhg)" opacity="0.95"/>
      <polyline points="10,22 15,22 18,15 21,29 24,18 27,22 35,22 38,22"
                fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div>
      <div class="navbar-title">GenGI</div>
      <div class="navbar-sub">General Genome Interpretation · MedFlow AI</div>
    </div>
  </div>
  <div>
    <span class="navbar-badge">Research Prototype</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Demo / fallback banner
    embedder, fell_back = load_embedder("demo" if demo_mode else "full", device=device)

    if demo_mode or fell_back:
        msg = "Demo mode — embeddings are deterministic random (MockEmbedder). Load a trained checkpoint for real predictions."
        if fell_back:
            msg = "⚡ Full model unavailable in this environment — automatically switched to Demo mode (MockEmbedder). Deploy locally with <code>pip install transformers accelerate</code> for real predictions."
        st.markdown(f"<div class='demo-banner'>⚠️ {msg}</div>", unsafe_allow_html=True)

    with st.spinner("Loading GenGI model …"):
        model = load_model()

    tab1, tab2, tab3 = st.tabs(["🔬 Variant Analysis", "🧑‍⚕️ Patient View", "📐 Architecture"])

    with tab1:
        tab_variant_analysis(demo_mode or fell_back, model, embedder)
    with tab2:
        tab_patient_view()
    with tab3:
        tab_architecture(demo_mode or fell_back)

    st.markdown("""
<div class="gengi-footer">
  Built by <strong>Dr. Mamadou Lamine TALL</strong> ·
  <a href="https://github.com/mamadoulaminetall">MedFlow AI</a> ·
  Powered by Nucleotide Transformer + PyTorch ·
  <a href="mailto:mamadoulaminetallgithub@gmail.com">mamadoulaminetallgithub@gmail.com</a>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
