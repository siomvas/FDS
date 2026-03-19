"""
generate_combined_report.py - Combined Cross-Dataset/Cross-Model Evaluation Report

Loads all partition eval results across datasets and feature models,
and generates a single HTML report with:
  - Coverage table (what's complete)
  - Per-dataset alpha sweep (JSD, FID, MMD, balance_cv)
  - Cross-model MMD comparison
  - VLM vs Dirichlet vs Random overview
  - Radar charts per dataset

Usage:
    python scripts/generate_combined_report.py \
        --output ./results/combined_report.html
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "Random":    "#636EFA",
    "Dirichlet": "#EF553B",
    "VLM":       "#00CC96",
}
MODEL_COLORS = {
    "clip":      "#F4A460",
    "dinov2":    "#8A2BE2",
    "inception": "#20B2AA",
}
MODEL_LABELS = {
    "clip":      "CLIP ViT-B/16",
    "dinov2":    "DINOv2-Base",
    "inception": "InceptionV3",
}
DATASET_LABELS = {
    "food101":       "Food-101",
    "tiny_imagenet": "Tiny ImageNet",
    "ade20k":        "ADE20K",
    "pascal_voc":    "Pascal VOC",
}

EVAL_BASE = "/path/to/gtai/FDS/vlm_eval"
MODELS    = ["clip", "dinov2", "inception"]
DATASETS  = ["food101", "tiny_imagenet", "ade20k", "pascal_voc"]

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            idx_path = Path(EVAL_BASE) / model / dataset / "results_index.json"
            if not idx_path.exists():
                continue
            idx = json.load(open(idx_path))
            for r in idx["rows"]:
                if r.get("status") != "success":
                    continue
                rp = Path(r["run_dir"]) / "result.json"
                if not rp.exists():
                    continue
                res = json.load(open(rp))
                m = res.get("metrics", {})
                s = res.get("summary", {})
                rows.append({
                    "feature_model": model,
                    "dataset": dataset,
                    "method": s.get("method", "?"),
                    "alpha": s.get("heterogeneity"),
                    "JSD": m.get("JSD_mean"),
                    "FID": m.get("fid_mean"),
                    "MMD": m.get("mmd_features_mean"),
                    "balance_cv": m.get("balance_cv"),
                    "wasserstein": m.get("wasserstein_mean"),
                    "vendi": m.get("vendi_score_mean"),
                    "knn_sep": m.get("knn_sep_mean"),
                    "n_clients": s.get("n_clients"),
                })
    return pd.DataFrame(rows)


# ── Plot Helpers ───────────────────────────────────────────────────────────────

def fig_to_html(fig, div_id: str) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)


def section(title: str, content: str, expanded: bool = False) -> str:
    state = "open" if expanded else ""
    return f"""
<details {state} class="section">
  <summary>{title}</summary>
  <div class="section-body">{content}</div>
</details>
"""


def subsection(title: str, content: str) -> str:
    return f'<div class="subsection"><h3>{title}</h3>{content}</div>'


# ── Coverage Table ─────────────────────────────────────────────────────────────

def coverage_table(df: pd.DataFrame) -> str:
    html = '<table class="coverage"><thead><tr><th>Dataset</th>'
    for m in MODELS:
        html += f"<th>{MODEL_LABELS[m]}</th>"
    html += "</tr></thead><tbody>"
    for d in DATASETS:
        html += f"<tr><td><b>{DATASET_LABELS[d]}</b></td>"
        for m in MODELS:
            sub = df[(df.dataset == d) & (df.feature_model == m)]
            n = len(sub)
            if n == 0:
                html += '<td class="cov-miss">✗ missing</td>'
            else:
                methods = sub["method"].unique().tolist()
                label = ", ".join(sorted(methods))
                html += f'<td class="cov-ok">✓ {n} partitions<br><small>{label}</small></td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html


# ── Alpha Sweep ────────────────────────────────────────────────────────────────

def fig_alpha_sweep(df: pd.DataFrame, dataset: str, metric: str, title: str) -> go.Figure:
    sub = df[(df.dataset == dataset) & df[metric].notna()]
    fig = go.Figure()

    for method, color in METHOD_COLORS.items():
        msub = sub[sub.method == method].sort_values("alpha")
        if msub.empty:
            continue
        if method == "Random":
            val = msub[metric].mean()
            fig.add_hline(y=val, line_dash="dash", line_color=color,
                          annotation_text="Random", annotation_position="right")
            continue
        # One line per feature model
        for fmodel in msub.feature_model.unique():
            fsub = msub[msub.feature_model == fmodel].sort_values("alpha")
            if fsub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=fsub["alpha"], y=fsub[metric],
                mode="lines+markers",
                name=f"{method} ({MODEL_LABELS.get(fmodel, fmodel)})",
                line=dict(color=color,
                          dash="solid" if fmodel == "clip" else
                               "dot"   if fmodel == "dinov2" else "dashdot"),
                marker=dict(size=7),
            ))

    fig.update_layout(
        title=title, xaxis_title="Alpha", yaxis_title=metric,
        height=350, margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Cross-Model MMD Comparison ─────────────────────────────────────────────────

def fig_cross_model_mmd(df: pd.DataFrame, dataset: str) -> go.Figure:
    sub = df[(df.dataset == dataset) & df["MMD"].notna() & (df.method != "Random")]
    if sub.empty:
        return None

    fig = go.Figure()
    for method, mcolor in METHOD_COLORS.items():
        msub = sub[sub.method == method]
        if msub.empty:
            continue
        for fmodel in MODELS:
            fsub = msub[msub.feature_model == fmodel].sort_values("alpha")
            if fsub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=fsub["alpha"], y=fsub["MMD"],
                mode="lines+markers",
                name=f"{method} / {MODEL_LABELS.get(fmodel, fmodel)}",
                line=dict(color=MODEL_COLORS.get(fmodel, "#888"),
                          dash="solid" if method == "Dirichlet" else "dot"),
                marker=dict(size=7),
            ))

    fig.update_layout(
        title=f"MMD by Feature Model — {DATASET_LABELS.get(dataset, dataset)}",
        xaxis_title="Alpha", yaxis_title="MMD",
        height=350, margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Method Comparison Bar (single metric, all alphas averaged) ─────────────────

def fig_method_bars(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    sub = df[df[metric].notna()]
    fig = go.Figure()

    for dataset in DATASETS:
        dsub = sub[sub.dataset == dataset]
        if dsub.empty:
            continue
        methods = ["Random", "Dirichlet", "VLM"]
        vals = []
        for method in methods:
            msub = dsub[dsub.method == method]
            vals.append(msub[metric].mean() if not msub.empty else None)
        fig.add_trace(go.Bar(
            name=DATASET_LABELS.get(dataset, dataset),
            x=methods, y=vals,
        ))

    fig.update_layout(
        title=title, barmode="group",
        height=380, margin=dict(t=40, b=40, l=50, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title=metric,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Balance CV ─────────────────────────────────────────────────────────────────

def fig_balance_cv(df: pd.DataFrame) -> go.Figure:
    sub = df[df["balance_cv"].notna()].copy()
    # Average balance_cv across feature models (it's the same)
    agg = sub.groupby(["dataset", "method", "alpha"])["balance_cv"].mean().reset_index()

    fig = go.Figure()
    for method, color in METHOD_COLORS.items():
        msub = agg[agg.method == method]
        if msub.empty:
            continue
        for dataset in DATASETS:
            dsub = msub[msub.dataset == dataset].sort_values("alpha")
            if dsub.empty:
                continue
            if method == "Random":
                continue  # always ~0, skip
            fig.add_trace(go.Scatter(
                x=dsub["alpha"], y=dsub["balance_cv"],
                mode="lines+markers",
                name=f"{method} / {DATASET_LABELS.get(dataset, dataset)}",
                line=dict(color=color,
                          dash="solid"   if dataset == "food101" else
                               "dot"     if dataset == "tiny_imagenet" else
                               "dashdot" if dataset == "ade20k" else "longdash"),
                marker=dict(size=7),
            ))

    fig.update_layout(
        title="Balance CV by Method & Dataset (lower = more balanced clients)",
        xaxis_title="Alpha", yaxis_title="Balance CV",
        height=400, margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Radar Chart ────────────────────────────────────────────────────────────────

def fig_radar(df: pd.DataFrame, dataset: str) -> Optional[go.Figure]:
    sub = df[df.dataset == dataset]
    metrics = ["JSD", "FID", "MMD", "balance_cv"]
    labels  = ["JSD", "FID", "MMD (×100)", "Balance CV"]

    # Normalise each metric 0-1 across all values in this dataset
    normed = sub.copy()
    for col in metrics:
        mn, mx = sub[col].min(), sub[col].max()
        if mx > mn:
            normed[col] = (sub[col] - mn) / (mx - mn)
        else:
            normed[col] = 0.0

    fig = go.Figure()
    for method, color in METHOD_COLORS.items():
        msub = normed[normed.method == method]
        if msub.empty:
            continue
        vals = [msub[c].mean() for c in metrics]
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels + [labels[0]],
            fill="toself", name=method,
            line=dict(color=color),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Normalised Metrics — {DATASET_LABELS.get(dataset, dataset)}",
        height=380, margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
    )
    return fig


# ── Summary Table ──────────────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{v:.4f}"

    agg = (df.groupby(["dataset", "method"])
             .agg(JSD=("JSD", "mean"), FID=("FID", "mean"),
                  MMD=("MMD", "mean"), balance_cv=("balance_cv", "mean"))
             .reset_index())

    html = """<table class="summary">
<thead><tr>
  <th>Dataset</th><th>Method</th>
  <th>JSD ↑het</th><th>FID ↑het</th><th>MMD ↑het</th><th>Balance CV ↑imbalance</th>
</tr></thead><tbody>"""

    for dataset in DATASETS:
        dsub = agg[agg.dataset == dataset]
        for i, (_, row) in enumerate(dsub.iterrows()):
            rowspan = f' rowspan="{len(dsub)}"' if i == 0 else ""
            dataset_cell = f'<td{rowspan}><b>{DATASET_LABELS.get(dataset, dataset)}</b></td>' if i == 0 else ""
            method = row["method"]
            html += f"""<tr>
  {dataset_cell}
  <td>{method}</td>
  <td>{fmt(row.JSD)}</td><td>{fmt(row.FID)}</td>
  <td>{fmt(row.MMD)}</td><td>{fmt(row.balance_cv)}</td>
</tr>"""

    html += "</tbody></table>"
    return html


# ── Main Report ────────────────────────────────────────────────────────────────

def generate_report(output_path: str):
    df = load_all_results()
    if df.empty:
        print("No results found.")
        return

    print(f"Loaded {len(df)} result rows across {df.dataset.nunique()} datasets, "
          f"{df.feature_model.nunique()} models, {df.method.nunique()} methods.")

    parts = []

    # ── Section 0: Coverage ──────────────────────────────────────────────────
    parts.append(section("Coverage", coverage_table(df), expanded=True))

    # ── Section 1: Summary Table ─────────────────────────────────────────────
    parts.append(section("Summary Table (all methods, averaged across alpha & feature model)",
                         summary_table(df)))

    # ── Section 2: Per-Dataset Alpha Sweeps ──────────────────────────────────
    sweep_html = ""
    for dataset in DATASETS:
        sub = df[df.dataset == dataset]
        if sub.empty:
            continue
        cols = []
        if sub["JSD"].notna().any():
            fig = fig_alpha_sweep(sub, dataset, "JSD",
                                  f"JSD vs Alpha — {DATASET_LABELS.get(dataset, dataset)}")
            cols.append(fig_to_html(fig, f"jsd_{dataset}"))
        if sub["FID"].notna().any():
            fig = fig_alpha_sweep(sub, dataset, "FID",
                                  f"FID vs Alpha — {DATASET_LABELS.get(dataset, dataset)}")
            cols.append(fig_to_html(fig, f"fid_{dataset}"))
        if sub["MMD"].notna().any():
            fig = fig_alpha_sweep(sub, dataset, "MMD",
                                  f"MMD vs Alpha — {DATASET_LABELS.get(dataset, dataset)}")
            cols.append(fig_to_html(fig, f"mmd_{dataset}"))

        grid = '<div class="grid-3">' + "".join(f'<div>{c}</div>' for c in cols) + '</div>'
        sweep_html += subsection(DATASET_LABELS.get(dataset, dataset), grid)

    parts.append(section("Alpha Sweep — Heterogeneity vs Metrics", sweep_html))

    # ── Section 3: Cross-Model MMD ───────────────────────────────────────────
    xmodel_html = ""
    for dataset in DATASETS:
        fig = fig_cross_model_mmd(df, dataset)
        if fig:
            xmodel_html += f'<div class="grid-item">{fig_to_html(fig, f"xmmd_{dataset}")}</div>'
    if xmodel_html:
        parts.append(section("Cross-Model MMD Agreement",
                             f'<div class="grid-2">{xmodel_html}</div>'))

    # ── Section 4: Balance CV ────────────────────────────────────────────────
    fig_bal = fig_balance_cv(df)
    parts.append(section("Client Size Imbalance (Balance CV)",
                         fig_to_html(fig_bal, "balance_cv_sweep")))

    # ── Section 5: Method Comparison Bars ────────────────────────────────────
    bar_html = ""
    for metric, title in [("JSD", "Average JSD by Method & Dataset"),
                           ("FID", "Average FID by Method & Dataset"),
                           ("MMD", "Average MMD by Method & Dataset")]:
        fig = fig_method_bars(df, metric, title)
        bar_html += f'<div class="grid-item">{fig_to_html(fig, f"bar_{metric}")}</div>'
    parts.append(section("Method Comparison (averaged across alpha)",
                         f'<div class="grid-3">{bar_html}</div>'))

    # ── Section 6: Radar Charts ───────────────────────────────────────────────
    radar_html = ""
    for dataset in DATASETS:
        fig = fig_radar(df, dataset)
        if fig:
            radar_html += f'<div class="grid-item">{fig_to_html(fig, f"radar_{dataset}")}</div>'
    if radar_html:
        parts.append(section("Normalised Radar Charts",
                             f'<div class="grid-2">{radar_html}</div>'))

    # ── Section 7: Raw Data ───────────────────────────────────────────────────
    cols_show = ["dataset", "feature_model", "method", "alpha",
                 "JSD", "FID", "MMD", "balance_cv", "wasserstein", "vendi", "knn_sep"]
    raw = df[[c for c in cols_show if c in df.columns]].copy()
    raw = raw.sort_values(["dataset", "method", "alpha", "feature_model"])
    for col in ["JSD", "FID", "MMD", "balance_cv", "wasserstein", "vendi", "knn_sep"]:
        if col in raw.columns:
            raw[col] = raw[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    parts.append(section("Raw Data", raw.to_html(index=False, classes="raw-table", border=0)))

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    plotlyjs = get_plotlyjs()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FDS Combined Evaluation Report</title>
<script>{plotlyjs}</script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto;
         padding: 20px; background: #f8f9fa; color: #333; }}
  h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  details.section {{ background: white; border-radius: 8px; margin: 12px 0;
                     box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  details.section > summary {{
    padding: 14px 20px; font-size: 1.1em; font-weight: 600;
    cursor: pointer; background: #f0f4f8; border-radius: 8px;
    list-style: none; display: flex; align-items: center; gap: 8px;
  }}
  details.section > summary::before {{ content: "▶"; font-size: 0.8em; transition: transform .2s; }}
  details.section[open] > summary::before {{ transform: rotate(90deg); }}
  .section-body {{ padding: 16px 20px; }}
  .subsection {{ margin-bottom: 20px; }}
  .subsection h3 {{ color: #555; border-left: 4px solid #3498db; padding-left: 10px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .grid-item {{ }}
  table.coverage {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  table.coverage th, table.coverage td {{ border: 1px solid #ddd; padding: 10px 14px;
                                          text-align: center; }}
  table.coverage th {{ background: #f0f4f8; font-weight: 600; }}
  td.cov-ok   {{ background: #d4edda; color: #155724; }}
  td.cov-miss {{ background: #f8d7da; color: #721c24; }}
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
  table.summary th {{ background: #f0f4f8; font-weight: 600; text-align: center; }}
  table.summary td:nth-child(1), table.summary td:nth-child(2) {{ text-align: left; }}
  table.raw-table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
  table.raw-table th, table.raw-table td {{ border: 1px solid #ddd; padding: 6px 10px; }}
  table.raw-table th {{ background: #f0f4f8; position: sticky; top: 0; }}
  p.note {{ color: #666; font-size: 0.85em; font-style: italic; }}
</style>
</head>
<body>
<h1>FDS — Combined Partition Evaluation Report</h1>
<p class="note">Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
Datasets: Food-101, Tiny ImageNet, ADE20K, Pascal VOC &nbsp;|&nbsp;
Feature models: CLIP ViT-B/16, DINOv2-Base, InceptionV3</p>
<p class="note"><b>Notes:</b>
FID always uses InceptionV3 internally (model-independent).
JSD is label-based (model-independent).
MMD uses the chosen feature model.
Pascal VOC JSD is fixed at ln(2) ≈ 0.693 due to multi-label detection format.
ADE20K/Pascal VOC have no Dirichlet partitions (VLM + Random only).
</p>
{''.join(parts)}
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/path/to/gtai/FDS/vlm_eval/combined_report.html")
    args = parser.parse_args()
    generate_report(args.output)
