"""Statistical summaries for canonical partition-evaluation artifacts."""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Sequence

import pandas as pd

from evaluation.eval_config import METRIC_META

SUMMARY_COLUMNS = [
    "metric",
    "category",
    "method",
    "count",
    "mean",
    "std",
    "sem",
    "ci95_low",
    "ci95_high",
    "higher_is_better",
]
RANKING_COLUMNS = SUMMARY_COLUMNS + ["rank", "preferred_direction"]
PAIRWISE_COLUMNS = [
    "metric",
    "category",
    "method_a",
    "method_b",
    "matched_runs",
    "mean_delta_method_b_minus_method_a",
    "median_delta_method_b_minus_method_a",
    "higher_is_better",
    "better_method",
]
MATCH_COLUMNS = ("dataset_name", "n_clients", "heterogeneity", "seed")


def available_metrics(df: pd.DataFrame) -> List[str]:
    """Return supported metric columns that are present and populated."""
    metrics: List[str] = []
    for metric in METRIC_META:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().any():
            metrics.append(metric)
    return metrics


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns or "method" not in df.columns:
        return _empty_frame(["method", metric])
    columns = ["method", metric, *[column for column in MATCH_COLUMNS if column in df.columns]]
    frame = df[columns].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame.dropna(subset=["method", metric])


def summarize_methods(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        frame = _metric_frame(df, metric)
        if frame.empty:
            continue
        meta = METRIC_META[metric]
        for method, series in frame.groupby("method", sort=False)[metric]:
            count = int(series.count())
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if count > 1 else 0.0
            sem = std / math.sqrt(count) if count > 1 else 0.0
            ci95 = 1.96 * sem if count > 1 else 0.0
            rows.append(
                {
                    "metric": metric,
                    "category": meta["category"],
                    "method": method,
                    "count": count,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - ci95,
                    "ci95_high": mean + ci95,
                    "higher_is_better": bool(meta["higher_is_better"]),
                }
            )
    if not rows:
        return _empty_frame(SUMMARY_COLUMNS)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def rank_methods(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    summary = summarize_methods(df, metrics)
    if summary.empty:
        return _empty_frame(RANKING_COLUMNS)

    rows = []
    for metric, group in summary.groupby("metric", sort=False):
        higher_is_better = bool(METRIC_META[metric]["higher_is_better"])
        ordered = group.sort_values("mean", ascending=not higher_is_better, kind="mergesort")
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            rows.append(
                {
                    **row._asdict(),
                    "rank": rank,
                    "preferred_direction": "higher" if higher_is_better else "lower",
                }
            )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def pairwise_method_deltas(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        frame = _metric_frame(df, metric)
        if frame.empty:
            continue

        match_columns = [column for column in MATCH_COLUMNS if column in frame.columns]
        if match_columns:
            pivot = frame.pivot_table(index=match_columns, columns="method", values=metric, aggfunc="mean")
        else:
            pivot = pd.DataFrame([frame.groupby("method", sort=False)[metric].mean().to_dict()])
        methods = list(pivot.columns)
        meta = METRIC_META[metric]
        higher_is_better = bool(meta["higher_is_better"])

        for method_a, method_b in combinations(methods, 2):
            pair = pivot[[method_a, method_b]].dropna()
            if pair.empty:
                continue
            delta = pair[method_b] - pair[method_a]
            mean_delta = float(delta.mean())
            if mean_delta == 0:
                better_method = "tie"
            elif higher_is_better:
                better_method = method_b if mean_delta > 0 else method_a
            else:
                better_method = method_b if mean_delta < 0 else method_a
            rows.append(
                {
                    "metric": metric,
                    "category": meta["category"],
                    "method_a": method_a,
                    "method_b": method_b,
                    "matched_runs": int(len(pair)),
                    "mean_delta_method_b_minus_method_a": mean_delta,
                    "median_delta_method_b_minus_method_a": float(delta.median()),
                    "higher_is_better": higher_is_better,
                    "better_method": better_method,
                }
            )
    if not rows:
        return _empty_frame(PAIRWISE_COLUMNS)
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
