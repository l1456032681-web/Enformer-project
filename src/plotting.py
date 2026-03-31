"""Plotting helpers for tracks and attribution."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


def _extract_preds(preds):
    if isinstance(preds, dict):
        if "human" in preds:
            return preds["human"]
        if "predictions" in preds:
            return preds["predictions"]
    return preds


def plot_tracks(
    preds,
    track_indices: Sequence[int],
    track_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    data = _extract_preds(preds)
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)

    if track_names is None:
        track_names = [f"track_{idx}" for idx in track_indices]

    fig, axes = plt.subplots(len(track_indices), 1, figsize=(12, 2.2 * len(track_indices)), sharex=True)
    if len(track_indices) == 1:
        axes = [axes]

    for ax, idx, name in zip(axes, track_indices, track_names):
        ax.plot(data[0, :, idx], lw=1.0)
        ax.set_ylabel(name)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Enformer bins")
    if title:
        fig.suptitle(title)
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
    y_label: Optional[str] = "Normalized signal (offset)",
    right_labels: bool = False,
    right_label_pad: float = 6.0,
    right_label_fontsize: float = 8.0,
):
    """Plot tracks in a compact waterfall (stacked) layout."""
    data = _extract_preds(preds)
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)

    if data.ndim == 3:
        data = data[0]

    if data.ndim != 2:
        raise ValueError(f"Expected data with shape (bins, tracks), got {data.shape}")

    if data.shape[1] != len(track_indices):
        data = data[:, track_indices]

    bins, n_tracks = data.shape
    if track_names is None:
        track_names = [f"track_{idx}" for idx in track_indices]

    if x_offset is None:
        x_offset = max(1.0, bins / (n_tracks * 4))

    x = np.arange(bins, dtype=float)
    colors = plt.get_cmap(cmap, n_tracks)

    fig, ax = plt.subplots(figsize=(11, 5))
    max_x = 0.0
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

        x_shift = x + i * x_offset
        y_shift = y * y_scale + i * y_offset
        ax.plot(x_shift, y_shift, lw=0.8, color=colors(i), alpha=0.9)
        max_x = max(max_x, float(x_shift[-1]))
        if right_labels and track_names:
            label_y = i * y_offset + 0.5 * y_scale
            ax.text(
                x_shift[-1] + right_label_pad,
                label_y,
                str(track_names[i]),
                color=colors(i),
                fontsize=right_label_fontsize,
                va="center",
                ha="left",
            )

    ax.set_xlabel("Enformer bins (offset per track)")
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    if right_labels and track_names:
        ax.set_xlim(right=max_x + right_label_pad * 10)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_attribution(
    attribution: np.ndarray,
    title: Optional[str] = None,
    cre_intervals: Optional[Iterable[tuple[int, int]]] = None,
):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(attribution, lw=0.8, color="#1f77b4")
    if cre_intervals:
        for start, end in cre_intervals:
            ax.axvspan(start, end, color="#ff7f0e", alpha=0.25)
    ax.set_xlabel("Input positions")
    ax.set_ylabel("Attribution")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def enformer_bin_genome_coords(
    window_start: int,
    bins: int = 896,
    bin_size: int = 128,
    input_size: int = 196_608,
) -> np.ndarray:
    """Map Enformer output bins to genomic coordinates."""
    output_span = bins * bin_size
    flank = (input_size - output_span) // 2
    output_start = window_start + flank
    centers = output_start + (np.arange(bins) * bin_size) + (bin_size / 2)
    return centers


def ccre_state_color(state: Optional[str]) -> str:
    if not state:
        return "#9e9e9e"
    state_upper = state.upper()
    if "PELS" in state_upper:
        return "#7b61ff"
    if "DELS" in state_upper:
        return "#9c27b0"
    if "DNASE-H3K4ME3" in state_upper or "PLS" in state_upper:
        return "#1b9e77"
    if "CTCF" in state_upper:
        return "#4f81bd"
    return "#9e9e9e"


def plot_locus_overlay(
    gene: str,
    chrom: str,
    window_start: int,
    window_end: int,
    track_values: np.ndarray,
    track_label: str,
    attribution: np.ndarray,
    ccre_intervals: Sequence,
    nearby_tss: Optional[Sequence[dict]] = None,
    predicted_peaks: Optional[Sequence[tuple[int, int]]] = None,
):
    """Plot a locus-style panel with track output, cCRE candidates, and attribution."""
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.35, 0.55, 1.15]},
    )

    gene_ax, track_ax, ccre_ax, attr_ax = axes
    x_track = enformer_bin_genome_coords(window_start, bins=len(track_values))
    x_attr = np.arange(window_start, window_end)

    gene_ax.set_ylim(0, 1)
    gene_ax.axis("off")
    gene_ax.text(
        0.0,
        1.02,
        f"{gene}",
        transform=gene_ax.transAxes,
        fontsize=16,
        fontstyle="italic",
        ha="left",
        va="bottom",
    )
    gene_ax.text(
        1.0,
        1.02,
        f"{chrom}:{window_start:,}-{window_end:,}",
        transform=gene_ax.transAxes,
        fontsize=11,
        ha="right",
        va="bottom",
    )
    gene_ax.hlines(0.45, window_start, window_end, color="#bdbdbd", lw=1.0)
    if nearby_tss:
        for item in nearby_tss:
            tss = item["tss"]
            color = "#111111" if item["gene"] == gene else "#757575"
            gene_ax.vlines(tss, 0.33, 0.57, color=color, lw=1.2)
            gene_ax.text(
                tss,
                0.62 if item["gene"] == gene else 0.72,
                item["gene"],
                color=color,
                fontsize=9 if item["gene"] == gene else 8,
                fontstyle="italic",
                ha="center",
                va="bottom",
            )

    wrapped_track_label = "\n".join(textwrap.wrap(track_label, width=28))

    track_ax.plot(x_track, track_values, color="#8b4fd1", lw=1.4)
    track_ax.fill_between(x_track, track_values, 0, color="#b89ae8", alpha=0.35)
    track_ax.set_ylabel(wrapped_track_label, rotation=0, ha="right", va="center", fontsize=9)
    track_ax.yaxis.set_label_coords(-0.10, 0.5)
    track_ax.spines[["top", "right"]].set_visible(False)

    ccre_ax.set_ylim(0, 1)
    ccre_ax.set_yticks([])
    ccre_ax.set_ylabel("Candidate", rotation=0, ha="right", va="center")
    ccre_ax.yaxis.set_label_coords(-0.09, 0.5)
    for interval in ccre_intervals:
        start = max(window_start, interval.start)
        end = min(window_end, interval.end)
        if end <= start:
            continue
        ccre_ax.add_patch(
            Rectangle(
                (start, 0.25),
                end - start,
                0.5,
                facecolor=ccre_state_color(getattr(interval, "state", None)),
                edgecolor="none",
                alpha=0.8,
            )
        )
    ccre_ax.spines[["top", "right", "left"]].set_visible(False)

    attr_ax.plot(x_attr, attribution, color="#b58900", lw=1.0)
    if predicted_peaks:
        for start, end in predicted_peaks:
            attr_ax.axvspan(start, end, color="#c8a100", alpha=0.18)
    attr_ax.set_ylabel("Grad x input", rotation=0, ha="right", va="center")
    attr_ax.yaxis.set_label_coords(-0.09, 0.5)
    attr_ax.spines[["top", "right"]].set_visible(False)
    attr_ax.set_xlabel("Genomic coordinate (Mb)")
    attr_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.2f}"))

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.08, hspace=0.18)
    return fig
