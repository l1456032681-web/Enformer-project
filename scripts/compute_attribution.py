#!/usr/bin/env python
"""Compute gradient x input attributions for selected genes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.attribution import call_peaks, gradient_x_input_for_track, reduce_attribution
from src.enformer import load_enformer, prepare_sequence_tensor
from src.genes import load_gene_coordinates
from src.genome import (
    ensure_uncompressed_fasta,
    fetch_sequence,
    get_fasta_reader,
    make_gene_window,
    pad_sequence,
    validate_sequence_length,
)
from src.plotting import plot_attribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords-csv", type=Path, required=True)
    parser.add_argument("--hg38", type=Path, required=True)
    parser.add_argument("--genes", type=str, default="")
    parser.add_argument("--track-index", type=int, required=True)
    parser.add_argument("--bin-index", type=int, default=-1)
    parser.add_argument("--head", type=str, default="human")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attribution"))
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--min-width", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coords = load_gene_coordinates(args.coords_csv)
    coord_map = {c.gene: c for c in coords}

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = list(coord_map.keys())

    if args.hg38.suffix == ".gz":
        fasta_out = args.hg38.with_suffix("")
    else:
        fasta_out = args.hg38
    fasta_path = ensure_uncompressed_fasta(args.hg38, fasta_out)
    fasta = get_fasta_reader(fasta_path)

    model = load_enformer(device=args.device)

    for gene in genes:
        coord = coord_map.get(gene)
        if coord is None:
            continue
        window = make_gene_window(coord.gene, coord.chrom, coord.tss)
        seq = fetch_sequence(fasta, window.chrom, window.start, window.end)
        if len(seq) != 196_608:
            seq = pad_sequence(seq, 196_608)
        validate_sequence_length(seq, 196_608)

        tokens = prepare_sequence_tensor(seq, device=args.device)
        with torch.no_grad():
            preds = model(tokens, head=args.head)
        if preds.ndim != 3:
            raise RuntimeError(f"Unexpected prediction shape: {preds.shape}")

        if args.bin_index >= 0:
            bin_index = args.bin_index
        else:
            bin_index = preds.shape[1] // 2

        attribution = gradient_x_input_for_track(
            model,
            tokens,
            track_index=args.track_index,
            bin_index=bin_index,
            head=args.head,
        )
        attr_1d = reduce_attribution(attribution, method="sum").squeeze(0)

        peaks = call_peaks(attr_1d, threshold_quantile=args.threshold_quantile, min_width=args.min_width)

        out_path = args.out_dir / f"{gene}_attr.npz"
        np.savez_compressed(
            out_path,
            attribution=attr_1d.detach().cpu().numpy(),
            peaks=np.asarray(peaks, dtype=np.int64),
            chrom=window.chrom,
            start=window.start,
            end=window.end,
            track_index=args.track_index,
            bin_index=bin_index,
        )

        if args.save_plots:
            fig = plot_attribution(attr_1d.detach().cpu().numpy(), title=gene)
            fig.savefig(args.out_dir / f"{gene}_attribution.png", dpi=150, bbox_inches="tight")

        # Save peaks as BED (absolute coordinates)
        if peaks:
            bed_path = args.out_dir / f"{gene}_peaks.bed"
            with bed_path.open("w") as handle:
                for idx, (start, end) in enumerate(peaks):
                    handle.write(f"{window.chrom}\t{window.start + start}\t{window.start + end}\t{gene}_peak{idx}\n")

        print(f"Saved {gene} attribution to {out_path}")


if __name__ == "__main__":
    main()
