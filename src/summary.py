"""Summaries for Enformer batch outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def summarize_outputs(outputs_dir: Path, track_selection_csv: Optional[Path] = None) -> pd.DataFrame:
    """Summarize per-gene track outputs saved as *_tracks.npz files."""
    rows: list[dict] = []

    track_names = None
    if track_selection_csv and track_selection_csv.exists():
        track_df = pd.read_csv(track_selection_csv)
        track_names = dict(zip(track_df["track_index"], track_df["track_name"]))

    for path in outputs_dir.glob("*_tracks.npz"):
        gene = path.stem.replace("_tracks", "")
        data = np.load(path)
        tracks = data["tracks"]  # shape: (1, bins, n_tracks)
        if tracks.ndim != 3:
            continue
        bins = tracks.shape[1]
        center = bins // 2

        mean_signal = tracks.mean(axis=1).squeeze(0)
        max_signal = tracks.max(axis=1).squeeze(0)
        max_bin = tracks.argmax(axis=1).squeeze(0)
        center_signal = tracks[:, center, :].squeeze(0)

        track_indices = data["track_indices"] if "track_indices" in data.files else np.arange(tracks.shape[2])
        for idx in range(tracks.shape[2]):
            track_index = int(track_indices[idx])
            rows.append(
                {
                    "gene": gene,
                    "track_index": track_index,
                    "track_name": track_names.get(track_index) if track_names else None,
                    "mean_signal": float(mean_signal[idx]),
                    "max_signal": float(max_signal[idx]),
                    "max_bin": int(max_bin[idx]),
                    "center_signal": float(center_signal[idx]),
                    "bins": bins,
                }
            )

    return pd.DataFrame(rows)


def write_summaries(outputs_dir: Path) -> tuple[Path, Path]:
    """Write per-gene and per-track summaries to CSV."""
    outputs_dir = Path(outputs_dir)
    track_selection = outputs_dir / "track_selection.csv"

    df = summarize_outputs(outputs_dir, track_selection_csv=track_selection)
    gene_path = outputs_dir / "gene_track_summary.csv"
    df.to_csv(gene_path, index=False)

    if not df.empty:
        agg = (
            df.groupby(["track_index", "track_name"], dropna=False)
            .agg(
                genes=("gene", "nunique"),
                mean_signal=("mean_signal", "mean"),
                max_signal=("max_signal", "mean"),
                center_signal=("center_signal", "mean"),
            )
            .reset_index()
        )
    else:
        agg = df

    track_path = outputs_dir / "track_summary.csv"
    agg.to_csv(track_path, index=False)

    return gene_path, track_path
