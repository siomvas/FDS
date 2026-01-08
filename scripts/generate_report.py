"""
generate_report.py - Interactive HTML Report Generator

Generates interactive HTML reports from partition evaluation results.
Uses Plotly for visualizations with hover, legend toggle, and export.

Usage:
    # Single dataset report
    python generate_report.py \
        --input ./results/tiny_imagenet/all_results.csv \
        --output ./results/tiny_imagenet/report.html

    # From directory of eval JSONs
    python generate_report.py \
        --input ./results/tiny_imagenet/ \
        --output ./results/tiny_imagenet/report.html
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Import metric metadata from shared config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluation.eval_config import METRIC_META

# ============================================================================
# Constants
# ============================================================================

METHOD_COLORS = {
    "Random": "#636EFA",           # Blue
    "Dirichlet": "#EF553B",        # Red
    "VLM": "#00CC96",              # Green
    "Embedding (best)": "#AB63FA", # Purple
    "Embedding": "#AB63FA",        # Purple (fallback)
}

# Priority metrics (shown prominently)
PRIMARY_METRICS = [
    "fid_mean",
    "id_mean", 
    "mmd_features_mean",
    "vendi_score_mean",
    "wasserstein_mean",
    "JSD_mean",
]

# Radar chart metrics
RADAR_METRICS = [
    "fid_mean",
    "mmd_features_mean", 
    "id_mean",
    "vendi_score_mean",
    "JSD_mean",
    "wasserstein_mean",
]

# Secondary metrics (in appendix)
APPENDIX_METRICS = [
    "balance_cv",
    "samples_per_client_min",
    "samples_per_client_max",
    "TV_mean",
    "centroid_dist_mean",
    "knn_sep_mean",
    "energy_dist_mean",
]

# Display names for metrics
METRIC_DISPLAY_NAMES = {
    "fid_mean": "FID",
    "id_mean": "Intrinsic Dim",
    "mmd_features_mean": "MMD",
    "vendi_score_mean": "Vendi Score",
    "wasserstein_mean": "Wasserstein",
    "JSD_mean": "JS Divergence",
    "JSD_global_mean": "JSD (Global)",
    "TV_mean": "Total Variation",
    "balance_cv": "Balance (CV)",
    "knn_sep_mean": "k-NN Separability",
    "centroid_dist_mean": "Centroid Distance",
    "energy_dist_mean": "Energy Distance",
    "feature_variance_mean": "Feature Variance",
}


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_results(input_path: Path) -> pd.DataFrame:
    """
    Load results from CSV or directory of JSON files.
    """
    if input_path.is_file() and input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.is_dir():
        # Load all *_eval.json files
        json_files = list(input_path.glob("*_eval.json"))
        if not json_files:
            raise ValueError(f"No *_eval.json files found in {input_path}")
        
        rows = []
        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)
                rows.append(data)
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Input must be a .csv file or directory: {input_path}")
    
    # Ensure method column exists
    if "method" not in df.columns:
        raise ValueError("DataFrame must have 'method' column")
    
    return df


def detect_available_metrics(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect which metrics are available (non-null) in the DataFrame.
    Returns dict with 'primary', 'radar', 'appendix' lists.
    """
    available = {
        "primary": [],
        "radar": [],
        "appendix": [],
        "all": []
    }
    
    for metric in PRIMARY_METRICS:
        if metric in df.columns and df[metric].notna().any():
            available["primary"].append(metric)
            available["all"].append(metric)
    
    for metric in RADAR_METRICS:
        if metric in df.columns and df[metric].notna().any():
            if metric not in available["radar"]:
                available["radar"].append(metric)
    
    for metric in APPENDIX_METRICS:
        if metric in df.columns and df[metric].notna().any():
            available["appendix"].append(metric)
            if metric not in available["all"]:
                available["all"].append(metric)
    
    return available


def cap_outliers(
    df: pd.DataFrame, 
    metrics: List[str], 
    percentile: float = 95
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Cap values at percentile, return capped df and notes about what was capped.
    """
    df_capped = df.copy()
    outlier_notes = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna()
        if len(values) == 0:
            continue
            
        cap_val = values.quantile(percentile / 100)
        outliers = df[metric] > cap_val
        
        if outliers.any():
            outlier_notes[metric] = {
                "cap_value": float(cap_val),
                "n_capped": int(outliers.sum()),
                "original_max": float(df[metric].max())
            }
            df_capped.loc[outliers, metric] = cap_val
    
    return df_capped, outlier_notes


def select_best_embedding(
    df: pd.DataFrame,
    primary_metrics: List[str]
) -> Tuple[pd.DataFrame, Dict[Tuple, pd.DataFrame]]:
    """
    For each (n_clients, heterogeneity) config:
    - Keep all non-Embedding rows as-is
    - Replace Embedding variants with single "Embedding (best)" row
    - Return original variants for expandable details
    """
    # Identify embedding rows (method == "Embedding" or starts with embedding-related terms)
    is_embedding = df["method"].str.contains("Embedding|DINOv2|CLIP|SigLIP", case=False, na=False)
    
    non_emb = df[~is_embedding].copy()
    emb = df[is_embedding].copy()
    
    if emb.empty:
        return df, {}
    
    best_rows = []
    variant_details = {}
    
    # Group by config
    group_cols = []
    if "n_clients" in emb.columns:
        group_cols.append("n_clients")
    if "heterogeneity" in emb.columns:
        group_cols.append("heterogeneity")
    
    if not group_cols:
        # No grouping possible, treat all as one group
        group_cols = None
        groups = [(None, emb)]
    else:
        groups = emb.groupby(group_cols)
    
    for key, group in groups:
        if len(group) == 0:
            continue
            
        # Compute rankings within this group
        rankings = compute_rankings_internal(group, primary_metrics)
        
        if rankings.empty:
            # Fallback: just pick first
            best_idx = group.index[0]
        else:
            best_idx = rankings["avg_rank"].idxmin()
        
        best_row = group.loc[best_idx].copy()
        
        # Store original method name as variant info
        original_method = best_row["method"]
        best_row["method"] = "Embedding (best)"
        best_row["_best_variant"] = original_method
        
        best_rows.append(best_row)
        
        # Store all variants for expandable section
        config_key = key if isinstance(key, tuple) else (key,)
        variant_details[config_key] = group
    
    if best_rows:
        best_emb_df = pd.DataFrame(best_rows)
        combined = pd.concat([non_emb, best_emb_df], ignore_index=True)
    else:
        combined = non_emb
    
    return combined, variant_details


# ============================================================================
# Ranking Logic
# ============================================================================

def compute_rankings_internal(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Internal ranking computation without groupby.
    Returns DataFrame with original index, metric ranks, and avg_rank.
    """
    if df.empty:
        return pd.DataFrame()
    
    rankings = pd.DataFrame(index=df.index)
    rankings["method"] = df["method"].values
    
    # Bug 1 fix: Preserve _best_variant if present
    if "_best_variant" in df.columns:
        rankings["_best_variant"] = df["_best_variant"].values
    
    valid_metrics = []
    for metric in metrics:
        if metric not in df.columns or df[metric].isna().all():
            continue
        
        meta = METRIC_META.get(metric, {})
        higher_is_better = meta.get("higher_is_better", False)
        ascending = not higher_is_better  # Lower rank = better
        
        rankings[f"{metric}_rank"] = df[metric].rank(ascending=ascending, method="min")
        valid_metrics.append(metric)
    
    if valid_metrics:
        rank_cols = [f"{m}_rank" for m in valid_metrics]
        rankings["avg_rank"] = rankings[rank_cols].mean(axis=1)
    else:
        rankings["avg_rank"] = np.nan
    
    return rankings


def compute_rankings(
    df: pd.DataFrame, 
    metrics: List[str],
    groupby: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute per-metric rankings, optionally within groups.
    Returns DataFrame with method, ranks per metric, and avg_rank.
    """
    if groupby is None:
        groupby = ["n_clients", "heterogeneity"]
    
    # Filter to valid groupby columns
    groupby = [col for col in groupby if col in df.columns]
    
    if not groupby:
        # No grouping, rank all together
        return compute_rankings_internal(df, metrics)
    
    # Rank within each group
    all_rankings = []
    for key, group in df.groupby(groupby):
        group_rankings = compute_rankings_internal(group, metrics)
        
        # Add group identifiers
        if isinstance(key, tuple):
            for i, col in enumerate(groupby):
                group_rankings[col] = key[i]
        else:
            group_rankings[groupby[0]] = key
        
        all_rankings.append(group_rankings)
    
    if all_rankings:
        return pd.concat(all_rankings, ignore_index=True)
    return pd.DataFrame()


def style_ranking_table(rankings_df: pd.DataFrame, metrics: List[str]) -> str:
    """
    Generate HTML table with conditional formatting.
    1st = gold, 2nd = silver, 3rd = bronze.
    Best avg_rank row highlighted green.
    """
    if rankings_df.empty:
        return "<p>No data available for ranking.</p>"
    
    # Prepare display DataFrame
    display_cols = ["method"]
    
    # Include best variant info if present (Bug 1 fix)
    if "_best_variant" in rankings_df.columns:
        display_cols.append("_best_variant")
    
    # Add group columns if present
    for col in ["n_clients", "heterogeneity"]:
        if col in rankings_df.columns:
            display_cols.append(col)
    
    # Add rank columns
    rank_cols = []
    for m in metrics:
        col = f"{m}_rank"
        if col in rankings_df.columns:
            rank_cols.append(col)
            display_cols.append(col)
    
    display_cols.append("avg_rank")
    
    # Filter to existing columns
    display_cols = [c for c in display_cols if c in rankings_df.columns]
    display_df = rankings_df[display_cols].copy()
    
    # Rename columns for display
    rename_map = {f"{m}_rank": METRIC_DISPLAY_NAMES.get(m, m) for m in metrics}
    rename_map["avg_rank"] = "Avg Rank"
    rename_map["method"] = "Method"
    rename_map["n_clients"] = "Clients"
    rename_map["heterogeneity"] = "α"
    rename_map["_best_variant"] = "Variant"  # Bug 1 fix: show winning variant
    display_df = display_df.rename(columns=rename_map)
    
    # Sort by avg rank
    if "Avg Rank" in display_df.columns:
        display_df = display_df.sort_values("Avg Rank")
    
    # Generate HTML with inline styles
    html = ['<table class="ranking-table">']
    
    # Header
    html.append('<thead><tr>')
    for col in display_df.columns:
        html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')
    
    # Body
    html.append('<tbody>')
    min_avg_rank = display_df["Avg Rank"].min() if "Avg Rank" in display_df.columns else None
    
    for idx, row in display_df.iterrows():
        # Highlight row with best avg rank
        row_style = ""
        if min_avg_rank is not None and row.get("Avg Rank") == min_avg_rank:
            row_style = ' class="best-row"'
        
        html.append(f'<tr{row_style}>')
        
        for col in display_df.columns:
            val = row[col]
            cell_style = ""
            
            # Apply medal styling to rank columns
            if col not in ["Method", "Clients", "α", "Avg Rank"] and pd.notna(val):
                if val == 1:
                    cell_style = ' class="rank-gold"'
                elif val == 2:
                    cell_style = ' class="rank-silver"'
                elif val == 3:
                    cell_style = ' class="rank-bronze"'
            
            # Format value
            if pd.isna(val):
                val_str = "—"
            elif isinstance(val, float):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            
            html.append(f'<td{cell_style}>{val_str}</td>')
        
        html.append('</tr>')
    
    html.append('</tbody></table>')
    
    return '\n'.join(html)


# ============================================================================
# Figure Generators
# ============================================================================

def get_method_color(method: str) -> str:
    """Get consistent color for method."""
    for key, color in METHOD_COLORS.items():
        if key in method:
            return color
    return "#999999"


def fig_grouped_bar(
    df: pd.DataFrame, 
    metric: str,
    title: Optional[str] = None,
    show_error: bool = True
) -> go.Figure:
    """
    Create grouped bar chart comparing methods for a single metric.
    Groups by (n_clients, heterogeneity) if multiple configs exist.
    """
    if metric not in df.columns:
        return go.Figure().add_annotation(text=f"Metric {metric} not available", showarrow=False)
    
    meta = METRIC_META.get(metric, {})
    std_field = meta.get("std_field")
    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    
    if title is None:
        title = meta.get("title", display_name)
    
    fig = go.Figure()
    
    methods = df["method"].unique()
    
    # Determine x-axis grouping
    has_n_clients = "n_clients" in df.columns and df["n_clients"].nunique() > 1
    has_alpha = "heterogeneity" in df.columns and df["heterogeneity"].nunique() > 1
    
    if has_n_clients or has_alpha:
        # Create x-axis labels from config
        df = df.copy()
        df["_config"] = ""
        if has_n_clients:
            df["_config"] += "n=" + df["n_clients"].astype(str)
        if has_n_clients and has_alpha:
            df["_config"] += ", "
        if has_alpha:
            df["_config"] += "α=" + df["heterogeneity"].astype(str)
        
        x_categories = df["_config"].unique()
        
        for method in methods:
            method_df = df[df["method"] == method]
            
            error_y = None
            if show_error and std_field and std_field in method_df.columns:
                error_y = dict(type="data", array=method_df[std_field].values, visible=True)
            
            fig.add_trace(go.Bar(
                name=method,
                x=method_df["_config"],
                y=method_df[metric],
                error_y=error_y,
                marker_color=get_method_color(method),
            ))
    else:
        # Simple bar chart
        for method in methods:
            method_df = df[df["method"] == method]
            val = method_df[metric].values[0] if len(method_df) > 0 else 0
            
            error_y = None
            if show_error and std_field and std_field in method_df.columns:
                std_val = method_df[std_field].values[0] if len(method_df) > 0 else 0
                error_y = dict(type="data", array=[std_val], visible=True)
            
            fig.add_trace(go.Bar(
                name=method,
                x=[method],
                y=[val],
                error_y=error_y,
                marker_color=get_method_color(method),
            ))
    
    fig.update_layout(
        title=title,
        yaxis_title=meta.get("ylabel", display_name),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    
    return fig


def fig_radar(
    df: pd.DataFrame,
    metrics: List[str],
    title: str = "Multi-Metric Comparison"
) -> go.Figure:
    """
    Radar chart comparing methods across normalized metrics.
    All axes point outward = better (flip 'lower is better' metrics).
    """
    # Filter to available metrics
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    
    if len(available) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough metrics for radar chart (need >= 3)", showarrow=False)
        return fig
    
    # Normalize 0-1
    norm_df = df.copy()
    for metric in available:
        col = df[metric]
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 1e-10:
            norm_df[metric] = (col - min_val) / (max_val - min_val)
        else:
            norm_df[metric] = 0.5
        
        # Flip so outward = better
        meta = METRIC_META.get(metric, {})
        if not meta.get("higher_is_better", False):
            norm_df[metric] = 1 - norm_df[metric]
    
    fig = go.Figure()
    
    methods = norm_df["method"].unique()
    theta_labels = [METRIC_DISPLAY_NAMES.get(m, m) for m in available]
    
    for method in methods:
        method_data = norm_df[norm_df["method"] == method]
        
        if len(method_data) > 1:
            # Average across configs
            values = [method_data[m].mean() for m in available]
        else:
            values = [method_data[m].values[0] for m in available]
        
        # Close the polygon
        values_closed = values + [values[0]]
        theta_closed = theta_labels + [theta_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=theta_closed,
            name=method,
            line_color=get_method_color(method),
            fill="toself",
            opacity=0.6,
        ))
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        template="plotly_white",
    )
    
    return fig


def fig_parallel_coordinates(
    df: pd.DataFrame,
    metrics: List[str],
    title: str = "Parallel Coordinates"
) -> go.Figure:
    """
    Parallel coordinates plot showing all metrics simultaneously.
    Color by method.
    """
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    
    if len(available) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough metrics for parallel coordinates", showarrow=False)
        return fig
    
    # Create method encoding for color
    methods = df["method"].unique()
    method_to_num = {m: i for i, m in enumerate(methods)}
    df = df.copy()
    df["_method_num"] = df["method"].map(method_to_num)
    
    # Build dimensions
    dimensions = []
    for metric in available:
        meta = METRIC_META.get(metric, {})
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        
        dimensions.append(dict(
            range=[df[metric].min(), df[metric].max()],
            label=label,
            values=df[metric],
        ))
    
    # Color scale matching method colors
    color_scale = []
    for i, method in enumerate(methods):
        color = get_method_color(method)
        color_scale.append([i / max(len(methods) - 1, 1), color])
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df["_method_num"],
            colorscale=color_scale if len(color_scale) > 1 else [[0, "#636EFA"], [1, "#636EFA"]],
            showscale=False,
        ),
        dimensions=dimensions,
    ))
    
    # Add legend manually via annotations
    legend_text = " | ".join([f'<span style="color:{get_method_color(m)}">{m}</span>' for m in methods])
    
    fig.update_layout(
        title=title,
        template="plotly_white",
    )
    
    return fig


def fig_to_html(fig: go.Figure, div_id: str) -> str:
    """Convert figure to HTML div with export buttons."""
    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": div_id,
            "height": 600,
            "width": 1000,
            "scale": 2
        },
        "displayModeBar": True,
        "modeBarButtonsToAdd": [],
        "displaylogo": False,
    }
    
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=config,
    )
    
    return html


# ============================================================================
# HTML Assembly
# ============================================================================

def render_accordion(title: str, content: str, expanded: bool = False) -> str:
    """Render an accordion section."""
    active_class = "active" if expanded else ""
    show_class = "show" if expanded else ""
    
    return f'''
    <button class="accordion {active_class}" onclick="toggleAccordion(this)">{title}</button>
    <div class="panel {show_class}">
        {content}
    </div>
    '''


def render_metric_dropdown(figures: Dict[str, str], default_metric: str) -> str:
    """Render a dropdown to switch between metric figures."""
    options = []
    for metric in figures.keys():
        display = METRIC_DISPLAY_NAMES.get(metric, metric)
        selected = "selected" if metric == default_metric else ""
        options.append(f'<option value="{metric}" {selected}>{display}</option>')
    
    # FIX: Use uuid instead of hash to prevent ID collisions across configs
    dropdown_id = f"dropdown_{uuid.uuid4().hex[:8]}"
    
    html = f'''
    <div class="metric-dropdown">
        <label>Select Metric: </label>
        <select id="{dropdown_id}" onchange="switchMetricFigure(this, '{dropdown_id}')">
            {''.join(options)}
        </select>
    </div>
    <div class="metric-figures" id="figures_{dropdown_id}">
    '''
    
    for metric, fig_html in figures.items():
        display = "block" if metric == default_metric else "none"
        html += f'<div class="metric-figure" data-metric="{metric}" style="display: {display}">{fig_html}</div>'
    
    html += '</div>'
    
    return html


def render_embedding_details(variant_details: Dict[Tuple, pd.DataFrame], metrics: List[str]) -> str:
    """Render expandable section showing all embedding variant details."""
    if not variant_details:
        return ""
    
    html = ['<div class="embedding-details">']
    html.append('<h4>Embedding Variant Details</h4>')
    
    for config_key, variants_df in variant_details.items():
        config_label = f"n={config_key[0]}" if len(config_key) > 0 else "all"
        if len(config_key) > 1:
            config_label += f", α={config_key[1]}"
        
        html.append(f'<h5>Config: {config_label}</h5>')
        
        # Simple table of variants
        display_cols = ["method"] + [m for m in metrics if m in variants_df.columns]
        display_df = variants_df[display_cols].copy()
        
        # Rename for display
        rename_map = {m: METRIC_DISPLAY_NAMES.get(m, m) for m in metrics}
        rename_map["method"] = "Variant"
        display_df = display_df.rename(columns=rename_map)
        
        html.append(display_df.to_html(index=False, classes="variant-table", float_format="%.3f"))
    
    html.append('</div>')
    
    return '\n'.join(html)


def render_config_filter(df: pd.DataFrame) -> Tuple[str, List[Tuple]]:
    """
    Render a dropdown filter for (n_clients, heterogeneity) configs.
    Returns filter HTML and list of config tuples.
    
    Bug 3 fix: This function is now actually used and creates a functional filter.
    """
    configs = []
    
    has_n = "n_clients" in df.columns and df["n_clients"].nunique() > 1
    has_alpha = "heterogeneity" in df.columns and df["heterogeneity"].nunique() > 1
    
    if has_n and has_alpha:
        configs = df.groupby(["n_clients", "heterogeneity"]).size().index.tolist()
    elif has_n:
        configs = [(n, None) for n in sorted(df["n_clients"].unique())]
    elif has_alpha:
        configs = [(None, a) for a in sorted(df["heterogeneity"].unique())]
    
    if len(configs) <= 1:
        return "", configs
    
    # Build dropdown
    options = ['<option value="all">All Configurations</option>']
    
    for i, config in enumerate(configs):
        if config[0] is not None and config[1] is not None:
            label = f"n={config[0]}, α={config[1]}"
        elif config[0] is not None:
            label = f"n={config[0]}"
        else:
            label = f"α={config[1]}"
        
        options.append(f'<option value="{i}">{label}</option>')
    
    filter_html = f'''
    <div class="config-filter" style="margin: 15px 0; padding: 10px; background: #e8f4f8; border-radius: 5px;">
        <label><strong>Filter by Configuration:</strong></label>
        <select id="config-filter" onchange="filterByConfig(this.value)" style="padding: 8px; margin-left: 10px;">
            {''.join(options)}
        </select>
    </div>
    '''
    
    return filter_html, configs


def generate_figures_per_config(
    df: pd.DataFrame,
    metrics: List[str],
    configs: List[Tuple],
    figure_prefix: str
) -> str:
    """
    Generate bar charts for each config, wrapped in filterable divs.
    
    Bug 3 fix: Instead of one unreadable chart, create separate charts per config.
    """
    if not configs or len(configs) <= 1:
        # Single config or no configs - just make one chart
        figures = {}
        for metric in metrics:
            fig = fig_grouped_bar(df, metric)
            figures[metric] = fig_to_html(fig, f"{figure_prefix}_{metric}")
        return render_metric_dropdown(figures, metrics[0])
    
    html_parts = []
    
    # Add "all configs" view (hidden by default if many configs)
    figures_all = {}
    for metric in metrics:
        fig = fig_grouped_bar(df, metric)
        figures_all[metric] = fig_to_html(fig, f"{figure_prefix}_all_{metric}")
    
    html_parts.append(f'<div class="config-view" data-config="all">')
    html_parts.append(render_metric_dropdown(figures_all, metrics[0]))
    html_parts.append('</div>')
    
    # Add per-config views
    for i, config in enumerate(configs):
        # Filter df to this config
        mask = pd.Series(True, index=df.index)
        if config[0] is not None:
            mask &= (df["n_clients"] == config[0])
        if config[1] is not None:
            mask &= (df["heterogeneity"] == config[1])
        
        df_filtered = df[mask]
        
        if df_filtered.empty:
            continue
        
        figures = {}
        for metric in metrics:
            fig = fig_grouped_bar(df_filtered, metric)
            figures[metric] = fig_to_html(fig, f"{figure_prefix}_cfg{i}_{metric}")
        
        html_parts.append(f'<div class="config-view" data-config="{i}" style="display: none;">')
        html_parts.append(render_metric_dropdown(figures, metrics[0]))
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def generate_summary_findings(df: pd.DataFrame, metrics: List[str]) -> str:
    """Generate key findings summary from data, grouped by config."""
    findings = []
    
    # Determine grouping columns
    group_cols = []
    if "n_clients" in df.columns and df["n_clients"].nunique() > 1:
        group_cols.append("n_clients")
    if "heterogeneity" in df.columns and df["heterogeneity"].nunique() > 1:
        group_cols.append("heterogeneity")
    
    # If no grouping needed, analyze globally but only for top metric
    if not group_cols:
        for metric in metrics[:1]:  # Just top priority metric
            if metric not in df.columns or df[metric].isna().all():
                continue
            
            meta = METRIC_META.get(metric, {})
            higher_is_better = meta.get("higher_is_better", False)
            display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            
            if higher_is_better:
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            
            best_method = df.loc[best_idx, "method"]
            best_val = df.loc[best_idx, metric]
            
            direction = "highest" if higher_is_better else "lowest"
            findings.append(f"<li><strong>{best_method}</strong> achieves {direction} {display_name}: {best_val:.3f}</li>")
    else:
        # Bug 2 fix: Find best method WITHIN each config
        for config_key, group in df.groupby(group_cols):
            if isinstance(config_key, tuple):
                config_label = ", ".join([f"{col}={val}" for col, val in zip(group_cols, config_key)])
            else:
                config_label = f"{group_cols[0]}={config_key}"
            
            # Find best method for primary metric within this config
            metric = metrics[0]  # Top priority metric
            if metric not in group.columns or group[metric].isna().all():
                continue
            
            meta = METRIC_META.get(metric, {})
            higher_is_better = meta.get("higher_is_better", False)
            display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            
            if higher_is_better:
                best_idx = group[metric].idxmax()
            else:
                best_idx = group[metric].idxmin()
            
            best_method = df.loc[best_idx, "method"]
            best_val = df.loc[best_idx, metric]
            
            direction = "highest" if higher_is_better else "lowest"
            findings.append(
                f"<li><strong>{config_label}</strong>: {best_method} achieves {direction} {display_name} ({best_val:.3f})</li>"
            )
    
    if findings:
        return f'<ul class="findings">{"".join(findings)}</ul>'
    return "<p>No findings available.</p>"


# ============================================================================
# Main Report Template
# ============================================================================

REPORT_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FDS Evaluation Report - {dataset}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ 
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        
        .meta {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        /* Accordion */
        .accordion {{
            background: #3498db;
            color: white;
            cursor: pointer;
            padding: 15px 20px;
            width: 100%;
            border: none;
            text-align: left;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            margin-top: 15px;
            transition: background 0.3s;
        }}
        .accordion:hover {{ background: #2980b9; }}
        .accordion::before {{ content: "▶ "; }}
        .accordion.active::before {{ content: "▼ "; }}
        .panel {{
            padding: 0;
            max-height: 0;
            overflow: hidden;
            background: white;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: max-height 0.3s ease-out, padding 0.3s;
        }}
        .panel.show {{
            max-height: none;
            padding: 20px;
        }}
        
        /* Tables */
        .ranking-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        .ranking-table th, .ranking-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .ranking-table th {{
            background: #34495e;
            color: white;
        }}
        .ranking-table tr:hover {{ background: #f5f5f5; }}
        .rank-gold {{ background: #FFD700 !important; font-weight: bold; }}
        .rank-silver {{ background: #C0C0C0 !important; }}
        .rank-bronze {{ background: #CD7F32 !important; color: white; }}
        .best-row {{ background: #90EE90 !important; }}
        
        .variant-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin: 10px 0;
        }}
        .variant-table th, .variant-table td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}
        .variant-table th {{ background: #ecf0f1; }}
        
        /* Metric dropdown */
        .metric-dropdown {{
            margin: 15px 0;
            padding: 10px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-dropdown select {{
            padding: 8px 15px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #bdc3c7;
        }}
        
        /* Figures */
        .figure-container {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        /* Findings */
        .findings {{
            background: #e8f6f3;
            padding: 15px 15px 15px 35px;
            border-left: 4px solid #1abc9c;
            border-radius: 0 8px 8px 0;
        }}
        .findings li {{ margin: 8px 0; }}
        
        /* Config tabs */
        .config-tabs {{
            margin: 15px 0;
        }}
        .tab-btn {{
            padding: 10px 20px;
            border: none;
            background: #ecf0f1;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .tab-btn:hover {{ background: #bdc3c7; }}
        .tab-btn.active {{
            background: #3498db;
            color: white;
        }}
        
        /* Embedding details */
        .embedding-details {{
            margin-top: 20px;
            padding: 15px;
            background: #fdf2e9;
            border-radius: 8px;
        }}
        
        /* Sortable table */
        .sortable th {{
            cursor: pointer;
            user-select: none;
        }}
        .sortable th:hover {{ background: #2c3e50; }}
        .sortable th::after {{ content: " ⇅"; opacity: 0.5; }}
        
        /* Print */
        @media print {{
            .accordion {{ background: #333; }}
            .panel {{ max-height: none !important; padding: 20px !important; }}
        }}
    </style>
</head>
<body>
    <h1>Federated Dataset Simulation - Evaluation Report</h1>
    
    <div class="meta">
        <strong>Dataset:</strong> {dataset} &nbsp;|&nbsp;
        <strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
        <strong>Partitions:</strong> {n_partitions} &nbsp;|&nbsp;
        <strong>Methods:</strong> {methods}
    </div>
    
    {content}
    
    <script>
        function toggleAccordion(el) {{
            el.classList.toggle("active");
            var panel = el.nextElementSibling;
            panel.classList.toggle("show");
        }}
        
        function switchMetricFigure(select, dropdownId) {{
            var container = document.getElementById("figures_" + dropdownId);
            var figures = container.getElementsByClassName("metric-figure");
            for (var i = 0; i < figures.length; i++) {{
                figures[i].style.display = "none";
            }}
            var selected = select.value;
            var target = container.querySelector('[data-metric="' + selected + '"]');
            if (target) target.style.display = "block";
        }}
        
        // Bug 3 fix: Working config filter function
        function filterByConfig(configValue) {{
            var views = document.querySelectorAll('.config-view');
            views.forEach(function(view) {{
                var viewConfig = view.getAttribute('data-config');
                if (configValue === 'all') {{
                    // Show only the "all" view
                    view.style.display = (viewConfig === 'all') ? 'block' : 'none';
                }} else {{
                    // Show only the selected config view
                    view.style.display = (viewConfig === configValue) ? 'block' : 'none';
                }}
            }});
        }}
        
        // Sortable tables
        document.querySelectorAll('.sortable th').forEach(function(th) {{
            th.addEventListener('click', function() {{
                var table = th.closest('table');
                var tbody = table.querySelector('tbody');
                var rows = Array.from(tbody.querySelectorAll('tr'));
                var idx = Array.from(th.parentElement.children).indexOf(th);
                var asc = th.dataset.sort !== 'asc';
                th.dataset.sort = asc ? 'asc' : 'desc';
                
                rows.sort(function(a, b) {{
                    var aVal = a.children[idx].textContent;
                    var bVal = b.children[idx].textContent;
                    var aNum = parseFloat(aVal);
                    var bNum = parseFloat(bVal);
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return asc ? aNum - bNum : bNum - aNum;
                    }}
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }});
                
                rows.forEach(function(row) {{ tbody.appendChild(row); }});
            }});
        }});
    </script>
</body>
</html>
'''


def generate_report(
    df: pd.DataFrame,
    output_path: Path,
    dataset_name: str = "Unknown"
) -> None:
    """
    Generate complete HTML report from evaluation results.
    """
    print(f"Generating report for {dataset_name}...")
    
    # Detect available metrics
    available = detect_available_metrics(df)
    print(f"  Available metrics: {len(available['all'])} total")
    
    # Handle embedding variants
    df_display, variant_details = select_best_embedding(df, available["primary"])
    
    # Cap outliers
    df_capped, outlier_notes = cap_outliers(df_display, available["all"])
    if outlier_notes:
        print(f"  Capped outliers in: {list(outlier_notes.keys())}")
    
    # Compute rankings
    rankings_df = compute_rankings(df_capped, available["primary"])
    
    # Build content sections
    content_parts = []
    
    # Bug 3 fix: Add config filter at top if multiple configs exist
    config_filter_html, configs = render_config_filter(df_capped)
    if config_filter_html:
        content_parts.append(config_filter_html)
    
    # === Section 1: Executive Summary ===
    summary_content = []
    summary_content.append("<h3>Method Rankings</h3>")
    summary_content.append(style_ranking_table(rankings_df, available["primary"]))
    summary_content.append("<h3>Key Findings</h3>")
    summary_content.append(generate_summary_findings(df_capped, available["primary"]))
    
    content_parts.append(render_accordion(
        "📊 Executive Summary",
        '\n'.join(summary_content),
        expanded=True
    ))
    
    # === Section 2: Feature Distance (Q1) ===
    distance_metrics = ["fid_mean", "mmd_features_mean", "wasserstein_mean", "energy_dist_mean"]
    distance_available = [m for m in distance_metrics if m in available["all"]]
    
    if distance_available:
        # Bug 3 fix: Generate per-config figures
        distance_content = generate_figures_per_config(
            df_capped, distance_available, configs, "distance"
        )
        content_parts.append(render_accordion(
            "🔬 Q1: How do methods compare? (Feature Distance)",
            f'<div class="figure-container">{distance_content}</div>',
            expanded=False
        ))
    
    # === Section 3: Feature Diversity (Q2) ===
    diversity_metrics = ["id_mean", "vendi_score_mean", "feature_variance_mean"]
    diversity_available = [m for m in diversity_metrics if m in available["all"]]
    
    if diversity_available:
        # Bug 3 fix: Generate per-config figures
        diversity_content = generate_figures_per_config(
            df_capped, diversity_available, configs, "diversity"
        )
        content_parts.append(render_accordion(
            "🌈 Q2: Feature Diversity Comparison",
            f'<div class="figure-container">{diversity_content}</div>',
            expanded=False
        ))
    
    # === Section 4: Label Heterogeneity (if applicable) ===
    label_metrics = ["JSD_mean", "JSD_global_mean", "TV_mean"]
    label_available = [m for m in label_metrics if m in available["all"]]
    
    if label_available:
        # Bug 3 fix: Generate per-config figures
        label_content = generate_figures_per_config(
            df_capped, label_available, configs, "label"
        )
        content_parts.append(render_accordion(
            "🏷️ Q3: Label Heterogeneity",
            f'<div class="figure-container">{label_content}</div>',
            expanded=False
        ))
    else:
        content_parts.append(render_accordion(
            "🏷️ Q3: Label Heterogeneity",
            '<p style="color: #999; font-style: italic;">Label metrics not available for this dataset.</p>',
            expanded=False
        ))
    
    # === Section 5: Multi-Metric Views ===
    multi_content = []
    
    # Radar chart
    if len(available["radar"]) >= 3:
        fig_radar_chart = fig_radar(df_capped, available["radar"])
        multi_content.append('<h3>Radar Chart (Normalized)</h3>')
        multi_content.append(f'<div class="figure-container">{fig_to_html(fig_radar_chart, "radar")}</div>')
    
    # Parallel coordinates
    if len(available["all"]) >= 2:
        fig_parallel = fig_parallel_coordinates(df_capped, available["all"][:8])  # Limit to 8 for readability
        multi_content.append('<h3>Parallel Coordinates</h3>')
        multi_content.append(f'<div class="figure-container">{fig_to_html(fig_parallel, "parallel")}</div>')
    
    if multi_content:
        content_parts.append(render_accordion(
            "📈 Multi-Metric Views",
            '\n'.join(multi_content),
            expanded=False
        ))
    
    # === Section 6: Embedding Variant Details ===
    if variant_details:
        # Fix: Cap outliers in variant details for consistency with main tables
        capped_variant_details = {}
        for config_key, variants_df in variant_details.items():
            capped_df, _ = cap_outliers(variants_df, available["all"])
            capped_variant_details[config_key] = capped_df
        
        emb_content = render_embedding_details(capped_variant_details, available["primary"])
        content_parts.append(render_accordion(
            "🔧 Embedding Variant Details",
            emb_content,
            expanded=False
        ))
    
    # === Section 7: Appendix (Secondary Metrics) ===
    if available["appendix"]:
        appendix_content = []
        
        figures = {}
        for metric in available["appendix"]:
            fig = fig_grouped_bar(df_capped, metric)
            figures[metric] = fig_to_html(fig, f"bar_appendix_{metric}")
        
        appendix_content.append(render_metric_dropdown(figures, available["appendix"][0]))
        
        content_parts.append(render_accordion(
            "📎 Appendix: Secondary Metrics",
            f'<div class="figure-container">{chr(10).join(appendix_content)}</div>',
            expanded=False
        ))
    
    # === Section 8: Raw Data Table ===
    # Select columns to display
    display_cols = ["method"]
    if "n_clients" in df_capped.columns:
        display_cols.append("n_clients")
    if "heterogeneity" in df_capped.columns:
        display_cols.append("heterogeneity")
    display_cols.extend([m for m in available["all"] if m in df_capped.columns])
    
    table_df = df_capped[display_cols].copy()
    
    # Rename columns
    rename_map = METRIC_DISPLAY_NAMES.copy()
    rename_map["method"] = "Method"
    rename_map["n_clients"] = "Clients"
    rename_map["heterogeneity"] = "α"
    table_df = table_df.rename(columns=rename_map)
    
    table_html = table_df.to_html(index=False, classes="ranking-table sortable", float_format="%.4f")
    
    content_parts.append(render_accordion(
        "📋 Raw Data Table (Sortable)",
        table_html,
        expanded=False
    ))
    
    # === Assemble Final HTML ===
    content = '\n'.join(content_parts)
    
    methods_list = ', '.join(df_display["method"].unique())
    
    html = REPORT_TEMPLATE.format(
        dataset=dataset_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_partitions=len(df),
        methods=methods_list,
        content=content,
    )
    
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Report saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From CSV
    python generate_report.py \\
        --input ./results/tiny_imagenet/all_results.csv \\
        --output ./results/tiny_imagenet/report.html \\
        --dataset tiny_imagenet

    # From directory of JSONs
    python generate_report.py \\
        --input ./results/tiny_imagenet/ \\
        --output ./results/tiny_imagenet/report.html \\
        --dataset tiny_imagenet
        """
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV or directory containing *_eval.json files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HTML file path")
    parser.add_argument("--dataset", type=str, default="Unknown",
                        help="Dataset name for report title")
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    df = load_results(input_path)
    print(f"Loaded {len(df)} evaluation results")
    
    # Generate report
    generate_report(
        df=df,
        output_path=Path(args.output),
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()