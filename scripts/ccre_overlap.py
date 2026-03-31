#!/usr/bin/env python
"""Compute overlap metrics between attribution peaks and ENCODE cCREs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.encode import (
    Interval,
    compute_precision_recall,
    filter_intervals,
    load_ccre_bed,
    nearest_interval_distance,
    overlap_size,
    overlapping_intervals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccre-bed", type=Path, required=True)
    parser.add_argument("--peaks-dir", type=Path, default=Path("outputs/attribution"))
    parser.add_argument("--gene-windows", type=Path, default=Path("outputs/gene_windows.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ccre_overlap.csv"))
    parser.add_argument("--peak-details-out", type=Path, default=None)
    parser.add_argument("--ccre-details-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ccre = load_ccre_bed(args.ccre_bed)
    if args.peak_details_out is None:
        args.peak_details_out = args.out.with_name(f"{args.out.stem}_peak_details.csv")
    if args.ccre_details_out is None:
        args.ccre_details_out = args.out.with_name(f"{args.out.stem}_ccre_details.csv")

    windows = pd.read_csv(args.gene_windows)
    rows = []
    peak_rows = []
    ccre_rows = []

    for _, row in windows.iterrows():
        gene = row["gene"]
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        peaks_path = args.peaks_dir / f"{gene}_peaks.bed"
        if not peaks_path.exists():
            continue

        predicted: list[Interval] = []
        with peaks_path.open() as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                parts = line.rstrip().split("\t")
                if len(parts) < 3:
                    continue
                p_chrom, p_start, p_end = parts[:3]
                peak_label = parts[3] if len(parts) > 3 else f"{gene}_peak{idx}"
                predicted.append(
                    Interval(chrom=p_chrom, start=int(p_start), end=int(p_end), label=peak_label)
                )

        truth = filter_intervals(ccre, chrom, start, end)
        precision, recall = compute_precision_recall(predicted, truth)

        rows.append(
            {
                "gene": gene,
                "precision": precision,
                "recall": recall,
                "predicted_peaks": len(predicted),
                "ccre_in_window": len(truth),
            }
        )

        for peak in predicted:
            matches = overlapping_intervals(peak, truth)
            nearest_distance = nearest_interval_distance(peak, truth)
            peak_rows.append(
                {
                    "gene": gene,
                    "peak_id": peak.label,
                    "chrom": peak.chrom,
                    "peak_start": peak.start,
                    "peak_end": peak.end,
                    "peak_width": peak.end - peak.start,
                    "hit": int(bool(matches)),
                    "matched_ccre_count": len(matches),
                    "matched_ccre_labels": ";".join(
                        interval.label or f"{interval.chrom}:{interval.start}-{interval.end}"
                        for interval in matches
                    ),
                    "nearest_ccre_distance": nearest_distance,
                    "max_overlap_bp": max(
                        (overlap_size(peak.start, peak.end, interval.start, interval.end) for interval in matches),
                        default=0,
                    ),
                }
            )

        for interval in truth:
            matches = overlapping_intervals(interval, predicted)
            nearest_distance = nearest_interval_distance(interval, predicted)
            ccre_rows.append(
                {
                    "gene": gene,
                    "ccre_id": interval.label or f"{interval.chrom}:{interval.start}-{interval.end}",
                    "ccre_state": interval.state,
                    "chrom": interval.chrom,
                    "ccre_start": interval.start,
                    "ccre_end": interval.end,
                    "ccre_width": interval.end - interval.start,
                    "hit": int(bool(matches)),
                    "matched_peak_count": len(matches),
                    "matched_peak_labels": ";".join(match.label or "" for match in matches),
                    "nearest_peak_distance": nearest_distance,
                    "max_overlap_bp": max(
                        (
                            overlap_size(interval.start, interval.end, match.start, match.end)
                            for match in matches
                        ),
                        default=0,
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(peak_rows).to_csv(args.peak_details_out, index=False)
    pd.DataFrame(ccre_rows).to_csv(args.ccre_details_out, index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.peak_details_out}")
    print(f"Wrote {args.ccre_details_out}")


if __name__ == "__main__":
    main()
