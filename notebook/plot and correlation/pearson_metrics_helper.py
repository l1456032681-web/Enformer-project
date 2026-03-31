from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def bin_centers_bp(n_bins: int, start: int, end: int) -> np.ndarray:
    edges = np.linspace(start, end, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def bin_centers_rel_kb(n_bins: int, start: int, end: int, tss: int) -> np.ndarray:
    return (bin_centers_bp(n_bins, start, end) - float(tss)) / 1000.0


def subset_tss_window(pred_signal: np.ndarray, exp_signal: np.ndarray, *, start: int, end: int, tss: int, tss_window_kb: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_signal = np.asarray(pred_signal, dtype=float)
    exp_signal = np.asarray(exp_signal, dtype=float)
    x_kb = bin_centers_rel_kb(len(pred_signal), start, end, tss)
    mask = np.abs(x_kb) <= float(tss_window_kb)
    return pred_signal[mask], exp_signal[mask], x_kb[mask]


def compute_tss_local_pearson_r(pred_signal: np.ndarray, exp_signal: np.ndarray, *, start: int, end: int, tss: int, tss_window_kb: float = 10.0) -> float:
    pred_sub, exp_sub, _ = subset_tss_window(pred_signal, exp_signal, start=start, end=end, tss=tss, tss_window_kb=tss_window_kb)
    return _safe_pearson(pred_sub, exp_sub)


def tss_bin_index(n_bins: int, start: int, end: int, tss: int) -> int:
    centers = bin_centers_bp(n_bins, start, end)
    return int(np.argmin(np.abs(centers - float(tss))))


def sum_three_tss_bins(signal: np.ndarray, *, start: int, end: int, tss: int) -> tuple[float, int, tuple[int, int]]:
    signal = np.asarray(signal, dtype=float)
    n_bins = len(signal)
    center = tss_bin_index(n_bins, start, end, tss)
    left = max(0, center - 1)
    right = min(n_bins, center + 2)
    return float(np.nansum(signal[left:right])), int(center), (left, right)


def zscore_ignore_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return np.full_like(x, np.nan, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0:
        return np.full_like(x, np.nan, dtype=float)
    out = (x - m) / s
    return out


def append_tss_local_metric_row(metric_rows: list[dict], *, gene: str, track_index: int, track_desc: str, encode_accession: str, encode_label: str, chrom: str, start: int, end: int, tss: int, pred_signal: np.ndarray, exp_signal: np.ndarray, tss_window_kb: float = 10.0) -> float:
    pred_sub, exp_sub, x_sub = subset_tss_window(pred_signal, exp_signal, start=start, end=end, tss=tss, tss_window_kb=tss_window_kb)
    pearson_r = _safe_pearson(pred_sub, exp_sub)
    metric_rows.append(
        {
            'gene': str(gene),
            'track_index': int(track_index),
            'track_desc': str(track_desc),
            'encode_accession': str(encode_accession),
            'encode_label': str(encode_label),
            'chrom': str(chrom),
            'start': int(start),
            'end': int(end),
            'tss': int(tss),
            'n_bins_full': int(len(pred_signal)),
            'n_bins_used': int(len(pred_sub)),
            'tss_window_kb': float(tss_window_kb),
            'pearson_r_tss_local': pearson_r,
            'x_kb_min': float(np.min(x_sub)) if len(x_sub) else np.nan,
            'x_kb_max': float(np.max(x_sub)) if len(x_sub) else np.nan,
        }
    )
    return pearson_r


def write_tss_local_outputs(metric_rows: Iterable[dict], outdir: Path) -> tuple[Path, Path] | tuple[None, None]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(metric_rows))
    if df.empty:
        return None, None
    per_gene_path = outdir / 'pearson_tss_local_per_gene_track.csv'
    df.to_csv(per_gene_path, index=False)
    summary_df = (
        df.groupby(['track_index', 'track_desc'], dropna=False)
        .agg(
            genes=('gene', 'nunique'),
            mean_pearson_r_tss_local=('pearson_r_tss_local', 'mean'),
            median_pearson_r_tss_local=('pearson_r_tss_local', 'median'),
            std_pearson_r_tss_local=('pearson_r_tss_local', 'std'),
        )
        .reset_index()
        .sort_values(['mean_pearson_r_tss_local', 'track_index'], ascending=[False, True])
    )
    summary_path = outdir / 'pearson_tss_local_summary_by_track.csv'
    summary_df.to_csv(summary_path, index=False)
    return per_gene_path, summary_path


def write_cage_across_genes_outputs(rows: Iterable[dict], outdir: Path, *, apply_log1p: bool = True, zscore_across_genes: bool = False) -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    per_gene_path = outdir / 'cage_three_bin_per_gene.csv'
    df.to_csv(per_gene_path, index=False)

    if df.empty:
        summary = pd.DataFrame([{'genes': 0, 'pearson_r_across_genes': np.nan, 'apply_log1p': apply_log1p, 'zscore_across_genes': zscore_across_genes}])
        summary_path = outdir / 'cage_three_bin_summary.csv'
        summary.to_csv(summary_path, index=False)
        return per_gene_path, summary_path

    pred = df['pred_three_bin_sum'].to_numpy(dtype=float)
    exp = df['exp_three_bin_sum'].to_numpy(dtype=float)

    if apply_log1p:
        pred = np.log1p(pred)
        exp = np.log1p(exp)

    if zscore_across_genes:
        pred = zscore_ignore_nan(pred)
        exp = zscore_ignore_nan(exp)

    pearson_r = _safe_pearson(pred, exp)
    summary = pd.DataFrame([
        {
            'genes': int(df['gene'].nunique()),
            'pearson_r_across_genes': pearson_r,
            'apply_log1p': bool(apply_log1p),
            'zscore_across_genes': bool(zscore_across_genes),
        }
    ])
    summary_path = outdir / 'cage_three_bin_summary.csv'
    summary.to_csv(summary_path, index=False)
    return per_gene_path, summary_path
