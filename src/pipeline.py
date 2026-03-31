"""Batch inference pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from src.enformer import predict_tracks, prepare_sequence_tensor
from src.genome import fetch_sequence, make_gene_window, normalize_chrom, pad_sequence, validate_sequence_length
from src.plotting import plot_tracks


@dataclass
class BatchResult:
    saved: list[str]
    skipped: list[str]
    missing: list[str]


def _load_done(state_path: Path, outputs_dir: Path) -> set[str]:
    done: set[str] = set()
    if state_path.exists():
        try:
            done = set(json.loads(state_path.read_text()).get("done", []))
        except json.JSONDecodeError:
            done = set()
    for path in outputs_dir.glob("*_tracks.npz"):
        done.add(path.stem.replace("_tracks", ""))
    return done


def run_batch_inference(
    genes: Sequence[str],
    coord_map,
    fasta,
    model,
    track_indices: Sequence[int],
    track_names: Optional[Sequence[str]] = None,
    output_dir: Path = Path("outputs"),
    window_size: int = 196_608,
    save_plots: bool = True,
    resume: bool = True,
    force: bool = False,
    state_path: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    device: str = "cuda",
    skip_chroms: Optional[Iterable[str]] = None,
) -> BatchResult:
    """Run Enformer inference over a list of genes and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if plots_dir is None:
        plots_dir = output_dir / "plots"
    plots_dir = Path(plots_dir)
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    if state_path is None:
        state_path = output_dir / "batch_state.json"
    state_path = Path(state_path)

    done = _load_done(state_path, output_dir) if resume else set()
    skip_chroms_norm = {normalize_chrom(c) for c in (skip_chroms or [])}

    saved: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    window_rows: list[dict] = []

    for gene in genes:
        if resume and not force and gene in done:
            skipped.append(gene)
            continue
        coord = coord_map.get(gene)
        if coord is None:
            missing.append(gene)
            continue
        if skip_chroms_norm and normalize_chrom(coord.chrom) in skip_chroms_norm:
            skipped.append(gene)
            continue

        window = make_gene_window(coord.gene, coord.chrom, coord.tss, window_size=window_size)
        window_rows.append(
            {
                "gene": coord.gene,
                "chrom": window.chrom,
                "tss": coord.tss,
                "start": window.start,
                "end": window.end,
                "strand": coord.strand,
            }
        )

        try:
            seq = fetch_sequence(fasta, window.chrom, window.start, window.end)
        except KeyError:
            missing.append(gene)
            continue

        seq = pad_sequence(seq, window_size)
        validate_sequence_length(seq, window_size)

        tokens = prepare_sequence_tensor(seq, device=device)
        preds = predict_tracks(model, tokens)
        preds_human = preds["human"] if isinstance(preds, dict) else preds
        subset = preds_human[:, :, track_indices].detach().cpu().numpy()
        np.savez_compressed(
            output_dir / f"{gene}_tracks.npz",
            tracks=subset,
            track_indices=np.asarray(track_indices, dtype=np.int64),
        )

        if save_plots:
            fig = plot_tracks(preds, track_indices, track_names, title=gene)
            fig.savefig(plots_dir / f"{gene}_tracks.png", dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)

        saved.append(gene)
        done.add(gene)
        if resume:
            state_path.write_text(json.dumps({"done": sorted(done)}, indent=2))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if window_rows:
        import pandas as pd

        pd.DataFrame(window_rows).to_csv(output_dir / "gene_windows.csv", index=False)

    if track_names:
        import pandas as pd

        pd.DataFrame(
            {"track_index": list(track_indices), "track_name": list(track_names)}
        ).to_csv(output_dir / "track_selection.csv", index=False)

    return BatchResult(saved=saved, skipped=skipped, missing=missing)
