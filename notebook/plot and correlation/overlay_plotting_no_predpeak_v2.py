"""Plotting helpers for muscle workflow and extension figures."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.encode import classify_ccre_label

COLOR_OUTPUT = "#2C7FB8"
COLOR_ATTR = "#225EA8"
COLOR_PRED = "#F28E2B"
COLOR_CCRE = "#BDBDBD"
COLOR_TEXT = "#333333"
COLOR_GRID = "#EAEAEA"
COLOR_SPINE = "#888888"
FIG_BG = "white"

CCRE_COLORS = {
    "promoter_like": "#4C78A8",
    "enhancer_like": "#59A14F",
    "ctcf_only": "#9C755F",
    "other": "#BDBDBD",
}


def _extract_preds(preds):
    if isinstance(preds, dict):
        if "human" in preds:
            return preds["human"]
        if "predictions" in preds:
            return preds["predictions"]
    return preds


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _style_axis(ax, hide_left: bool = False, grid_y: bool = False):
    ax.set_facecolor(FIG_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color(COLOR_SPINE)
        ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=COLOR_TEXT, labelsize=10)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_TEXT)
    if grid_y:
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.8)


def _make_rel_kb(x_coords: np.ndarray, tss: int) -> np.ndarray:
    return (np.asarray(x_coords, dtype=float) - float(tss)) / 1000.0


def _format_ticks_kb(ax, ticks_kb: np.ndarray):
    ax.set_xticks(ticks_kb)
    labels = []
    for t in ticks_kb:
        if abs(t) < 1e-9:
            labels.append("0")
        elif float(t).is_integer():
            labels.append(f"{int(t)}")
        else:
            labels.append(f"{t:.1f}")
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
        label.set_color(COLOR_TEXT)


def _format_ticks_bp(ax, ticks_bp: np.ndarray):
    ax.set_xticks(ticks_bp)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
        label.set_color(COLOR_TEXT)


def _draw_interval_band(ax, intervals, y0, height, color, alpha=0.7, x_transform=None):
    if not intervals:
        return
    for interval in intervals:
        start = interval.start
        end = interval.end
        if x_transform is not None:
            start = x_transform(start)
            end = x_transform(end)
        width = max(1e-6, end - start)
        ax.add_patch(Rectangle((start, y0), width, height, facecolor=color, edgecolor="none", alpha=alpha))


def _draw_ccre_band(ax, intervals, y0, height, x_transform=None):
    if not intervals:
        return
    for interval in intervals:
        label_type = classify_ccre_label(getattr(interval, "label", None))
        color = CCRE_COLORS.get(label_type, COLOR_CCRE)
        start = interval.start
        end = interval.end
        if x_transform is not None:
            start = x_transform(start)
            end = x_transform(end)
        width = max(1e-6, end - start)
        ax.add_patch(Rectangle((start, y0), width, height, facecolor=color, edgecolor="none", alpha=0.90))

def plot_tracks(
    preds,
    track_indices,
    track_names=None,
    title=None,
    normalize="none",
):
    """
    Simple multi-track plotting function used by src.pipeline.run_batch_inference.
    Returns a matplotlib Figure.
    """
    data = _extract_preds(preds)
    data = _to_numpy(data)

    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected data with shape (bins, tracks), got {data.shape}")

    # If preds contains all tracks, subset to the requested indices
    if data.shape[1] != len(track_indices):
        data = data[:, track_indices]

    bins, n_tracks = data.shape

    if track_names is None:
        track_names = [f"track_{idx}" for idx in track_indices]

    x = np.arange(bins, dtype=float)

    fig_h = max(2.2 * n_tracks, 4.0)
    fig, axes = plt.subplots(
        n_tracks,
        1,
        figsize=(12, fig_h),
        sharex=True,
        facecolor=FIG_BG,
        constrained_layout=False,
    )

    if n_tracks == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        y = data[:, i].astype(float)

        if normalize == "minmax":
            min_v = float(np.min(y))
            max_v = float(np.max(y))
            span = max_v - min_v
            y = (y - min_v) / span if span > 0 else np.zeros_like(y)
        elif normalize == "zscore":
            mean = float(np.mean(y))
            std = float(np.std(y))
            y = (y - mean) / std if std > 0 else np.zeros_like(y)
        elif normalize != "none":
            raise ValueError(f"Unknown normalization: {normalize}")

        ax.plot(x, y, lw=1.1, color=COLOR_OUTPUT)
        ax.set_ylabel(str(track_names[i]), fontsize=9, color=COLOR_TEXT)
        _style_axis(ax, grid_y=True)

        if i < n_tracks - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Enformer bins", color=COLOR_TEXT)

    if title:
        fig.suptitle(str(title), fontsize=14, color=COLOR_TEXT, y=0.995)

    fig.tight_layout()
    return fig

def plot_tracks_waterfall(
    preds,
    track_indices: Sequence[int],
    track_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    normalize: str = "minmax",
    x_offset: Optional[float] = None,
    y_offset: float = 1.0,
    y_scale: float = 1.0,
    cmap: str = "viridis",
    y_label: Optional[str] = "Normalized signal (a.u.)",
    right_labels: bool = False,
    right_label_pad: float = 0.8,
    right_label_fontsize: float = 8.0,
    x_coords: Optional[Sequence[float]] = None,
    tss: Optional[int] = None,
    relative_kb: bool = False,
    relative_to_tss: Optional[bool] = None,
):
    data = _extract_preds(preds)
    data = _to_numpy(data)

    if relative_to_tss is not None:
        relative_kb = bool(relative_to_tss)

    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected data with shape (bins, tracks), got {data.shape}")

    if data.shape[1] != len(track_indices):
        data = data[:, track_indices]

    bins, n_tracks = data.shape
    if track_names is None:
        track_names = [f"track_{idx}" for idx in track_indices]

    if x_coords is None:
        x_base = np.arange(bins, dtype=float)
        x_label = "Enformer bins"
        use_relative = False
    else:
        x_base = np.asarray(x_coords, dtype=float)
        use_relative = bool(relative_kb and (tss is not None))
        if use_relative:
            x_base = _make_rel_kb(x_base, int(tss))
            x_label = "Position relative to TSS (kb)"
        else:
            x_label = "Genomic coordinate (hg38)"

    colors = plt.get_cmap(cmap, n_tracks)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=FIG_BG)
    max_x = float(np.max(x_base))
    min_x = float(np.min(x_base))

    for i in range(n_tracks):
        y = data[:, i].astype(float)
        if normalize == "minmax":
            min_v = float(np.min(y))
            max_v = float(np.max(y))
            span = max_v - min_v
            y = (y - min_v) / span if span > 0 else np.zeros_like(y)
        elif normalize == "zscore":
            mean = float(np.mean(y))
            std = float(np.std(y))
            y = (y - mean) / std if std > 0 else np.zeros_like(y)
        elif normalize != "none":
            raise ValueError(f"Unknown normalization: {normalize}")

        y_shift = y * y_scale + i * y_offset
        ax.plot(x_base, y_shift, lw=1.0, color=colors(i), alpha=0.95)
        if right_labels and track_names:
            label_y = i * y_offset + 0.5 * y_scale
            ax.text(max_x + right_label_pad, label_y, str(track_names[i]), color=colors(i), fontsize=right_label_fontsize, va="center", ha="left")

    if use_relative:
        ax.axvline(0.0, linestyle="--", linewidth=0.9, color="#999999", alpha=0.8)
        ticks = np.linspace(min_x, max_x, 6)
        _format_ticks_kb(ax, ticks)
    else:
        ticks = np.linspace(min_x, max_x, 6)
        _format_ticks_bp(ax, ticks)

    ax.set_xlabel(x_label, color=COLOR_TEXT)
    if y_label:
        ax.set_ylabel(y_label, color=COLOR_TEXT)
    ax.set_yticks([])
    _style_axis(ax, hide_left=False, grid_y=False)
    if right_labels and track_names:
        ax.set_xlim(min_x, max_x + right_label_pad * 8)
    if title:
        ax.set_title(title, color=COLOR_TEXT, fontsize=14, pad=8)
    fig.tight_layout()
    return fig


def plot_gene_overlay(
    gene: str,
    chrom: str,
    window_start: Optional[int] = None,
    tss: Optional[int] = None,
    bin_starts: Optional[np.ndarray] = None,
    pred_signal: Optional[np.ndarray] = None,
    attribution_scores: Optional[np.ndarray] = None,
    candidate_intervals=None,
    predicted_intervals=None,
    track_name: str | None = None,
):
    if bin_starts is None or pred_signal is None or attribution_scores is None:
        raise ValueError("bin_starts, pred_signal, and attribution_scores are required")

    bin_starts = np.asarray(bin_starts, dtype=float)
    pred_signal = np.asarray(pred_signal, dtype=float)
    attribution_scores = np.asarray(attribution_scores, dtype=float)

    if tss is None:
        if window_start is None:
            raise ValueError("Either tss or window_start must be provided")
        x_output = bin_starts
        x_attr = np.arange(len(attribution_scores), dtype=float) + float(window_start)
        x_transform = None
        x_label = "Genomic coordinate (hg38)"
        use_relative = False
        xmin = float(np.min(x_output)) - 5.0
        xmax = float(np.max(x_output)) + 5.0
        ticks = np.linspace(xmin, xmax, 6)
    else:
        if window_start is None:
            raise ValueError("window_start is required when plotting relative to TSS")
        x_output = _make_rel_kb(bin_starts, int(tss))
        x_attr_bp = np.arange(len(attribution_scores), dtype=float) + float(window_start)
        x_attr = _make_rel_kb(x_attr_bp, int(tss))
        x_transform = lambda x, t=float(tss): (x - t) / 1000.0
        x_label = "Position relative to TSS (kb)"
        use_relative = True
        xmin = float(np.min(x_output)) - 5.0
        xmax = float(np.max(x_output)) + 5.0
        ticks = np.linspace(xmin, xmax, 6)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 8.4),
        sharex=False,
        gridspec_kw={"height_ratios": [1.35, 1.35, 0.42], "hspace": 0.10},
        facecolor=FIG_BG,
    )

    ax = axes[0]
    ax.plot(x_output, pred_signal, lw=1.5, color=COLOR_OUTPUT)
    ax.set_ylabel("Enformer output", fontsize=11, color=COLOR_TEXT)
    ax.set_title(f"{gene} | {chrom} | {track_name or 'selected track'}", fontsize=14, color=COLOR_TEXT, pad=10)
    _style_axis(ax, grid_y=True)
    if use_relative:
        ax.axvline(0.0, linestyle="--", linewidth=0.9, color="#999999", alpha=0.8)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis="x", labelbottom=False)

    ax = axes[1]
    ax.plot(x_attr, attribution_scores, lw=1.0, color=COLOR_ATTR)
    ax.set_ylabel("Gradient × input", fontsize=11, color=COLOR_TEXT)
    _style_axis(ax, grid_y=True)
    if use_relative:
        ax.axvline(0.0, linestyle="--", linewidth=0.9, color="#999999", alpha=0.8)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis="x", labelbottom=False)

    ax = axes[2]
    _draw_ccre_band(ax, candidate_intervals, y0=0.18, height=0.64, x_transform=x_transform)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("ENCODE\ncCRE", rotation=0, ha="right", va="center", labelpad=24, fontsize=10, color=COLOR_TEXT)
    ax.set_xlabel(x_label, fontsize=11, color=COLOR_TEXT)
    _style_axis(ax, hide_left=True)
    if use_relative:
        ax.axvline(0.0, linestyle="--", linewidth=0.9, color="#999999", alpha=0.8)
    ax.set_xlim(xmin, xmax)

    if use_relative:
        _format_ticks_kb(ax, ticks)
    else:
        _format_ticks_bp(ax, ticks)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.08, hspace=0.12)
    return fig


def plot_four_gene_overview(gene_panels: list[dict]):
    n = len(gene_panels)
    if n < 1:
        raise ValueError("gene_panels must contain at least one gene")

    rows_per_gene = 3
    total_rows = n * rows_per_gene
    fig, axes = plt.subplots(
        total_rows,
        1,
        figsize=(18, max(6.0, 4.2 * n)),
        sharex=False,
        gridspec_kw={"height_ratios": [1.20, 1.00, 0.34] * n, "hspace": 0.18},
        facecolor=FIG_BG,
        constrained_layout=False,
    )

    if total_rows == 1:
        axes = [axes]

    for i, panel in enumerate(gene_panels):
        base = i * rows_per_gene
        gene = panel["gene"]
        track_name = panel["track_name"]
        tss = panel.get("tss")

        if "x_output" in panel:
            x_output = np.asarray(panel["x_output"], dtype=float)
            use_relative = True
        else:
            bin_starts = np.asarray(panel["bin_starts"], dtype=float)
            x_output = _make_rel_kb(bin_starts, int(tss)) if tss is not None else bin_starts
            use_relative = tss is not None

        if "x_attr_rel" in panel:
            x_attr_rel = np.asarray(panel["x_attr_rel"], dtype=float)
            use_relative = True
        else:
            x_attr = np.asarray(panel["x_attr"], dtype=float)
            x_attr_rel = _make_rel_kb(x_attr, int(tss)) if tss is not None else x_attr

        pred_signal = np.asarray(panel["pred_signal"], dtype=float)
        attr_scores = np.asarray(panel["attr_scores"], dtype=float)
        candidate_intervals = panel.get("candidate_intervals", [])

        if use_relative:
            xmin = float(np.min(x_attr_rel))
            xmax = float(np.max(x_attr_rel))
            x_transform = None
            if tss is not None and candidate_intervals and getattr(candidate_intervals[0], "start", None) is not None and candidate_intervals[0].start > 1000:
                x_transform = lambda x, t=float(tss): (x - t) / 1000.0
            ticks = np.linspace(xmin, xmax, 5)
            x_label = "Position relative to TSS (kb)"
        else:
            xmin = float(np.min(x_attr_rel))
            xmax = float(np.max(x_attr_rel))
            x_transform = None
            ticks = np.linspace(xmin, xmax, 5)
            x_label = "Genomic coordinate (hg38)"

        ax = axes[base]
        ax.plot(x_output, pred_signal, lw=1.2, color=COLOR_OUTPUT)
        ax.set_ylabel("Output", fontsize=10, color=COLOR_TEXT)
        ax.set_title(f"{gene} | {track_name}", fontsize=13, color=COLOR_TEXT, pad=8, y=1.02)
        _style_axis(ax, grid_y=True)
        if use_relative:
            ax.axvline(0.0, linestyle="--", linewidth=0.8, color="#999999", alpha=0.8)
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis="x", labelbottom=False)

        ax = axes[base + 1]
        ax.plot(x_attr_rel, attr_scores, lw=0.95, color=COLOR_ATTR)
        ax.set_ylabel("G×I", fontsize=10, color=COLOR_TEXT)
        _style_axis(ax, grid_y=True)
        if use_relative:
            ax.axvline(0.0, linestyle="--", linewidth=0.8, color="#999999", alpha=0.8)
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis="x", labelbottom=False)

        ax = axes[base + 2]
        _draw_ccre_band(ax, candidate_intervals, y0=0.2, height=0.6, x_transform=x_transform)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("cCRE", rotation=0, ha="right", va="center", labelpad=18, fontsize=9, color=COLOR_TEXT)
        ax.set_xlabel(x_label, fontsize=10, color=COLOR_TEXT)
        _style_axis(ax, hide_left=True)
        if use_relative:
            ax.axvline(0.0, linestyle="--", linewidth=0.8, color="#999999", alpha=0.8)
        ax.set_xlim(xmin, xmax)

        if use_relative:
            _format_ticks_kb(ax, ticks)
        else:
            _format_ticks_bp(ax, ticks)

    for idx, ax in enumerate(axes):
        if idx % rows_per_gene != rows_per_gene - 1:
            ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.06, hspace=0.22)
    return fig
