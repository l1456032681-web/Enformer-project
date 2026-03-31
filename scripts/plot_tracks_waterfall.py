#!/usr/bin/env python
"""Create compact waterfall plots for saved track outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from src.plotting import plot_tracks_waterfall


def shorten_track_label(name: str) -> str:
    """Compress ENCODE-style descriptions into compact assay:tissue labels."""
    raw = str(name)
    assay = "Track"
    detail = raw

    if raw.startswith("DNASE:"):
        assay = "DNase"
        detail = raw[len("DNASE:") :]
    elif raw.startswith("CHIP:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            assay = parts[1]
            detail = parts[2]
        else:
            detail = raw[len("CHIP:") :]

    replacements = [
        ("myotube originated from skeletal muscle myoblast", "myotube"),
        ("skeletal muscle myoblast", "skel myoblast"),
        ("skeletal muscle cell", "skel muscle cell"),
        ("psoas muscle male adult (27 years) and male adult (35 years)", "psoas muscle"),
        ("smooth muscle cell of the brain vasculature female", "smooth muscle"),
        ("cardiac muscle cell", "cardiac muscle"),
        ("heart male adult (27 years) and male adult (35 years)", "heart"),
        ("cardiac mesoderm", "cardiac mesoderm"),
        ("male adult (22 years)", ""),
        ("male adult (27 years) and male adult (35 years)", ""),
        ("female", ""),
    ]
    for old, new in replacements:
        detail = detail.replace(old, new)

    detail = " ".join(detail.replace("(", " ").replace(")", " ").split(" "))
    detail = " ".join(detail.split())
    label = f"{assay}: {detail}".strip()
    return label[:38]


def uniquify_labels(labels: list[str], indices: np.ndarray) -> list[str]:
    """Append track index when multiple selected tracks collapse to the same short label."""
    counts = Counter(labels)
    result: list[str] = []
    for label, idx in zip(labels, indices):
        if counts[label] > 1:
            result.append(f"{label} [{int(idx)}]")
        else:
            result.append(label)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--genes", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "zscore", "none"])
    parser.add_argument("--x-offset", type=float, default=None)
    parser.add_argument("--y-offset", type=float, default=1.0)
    parser.add_argument("--y-scale", type=float, default=1.0)
    parser.add_argument("--y-label", type=str, default="Normalized signal (offset)")
    parser.add_argument("--right-labels", action="store_true")
    parser.add_argument("--suffix", type=str, default="_tracks_waterfall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    track_names = None
    track_map = {}
    selection_path = outputs_dir / "track_selection.csv"
    if selection_path.exists():
        with selection_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_idx = row.get("track_index")
                track_name = row.get("track_name")
                if track_idx is None or track_name is None:
                    continue
                track_map[int(track_idx)] = track_name

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = [p.stem.replace("_tracks", "") for p in outputs_dir.glob("*_tracks.npz")]

    for gene in genes:
        path = outputs_dir / f"{gene}_tracks.npz"
        if not path.exists():
            print(f"Missing tracks for {gene}: {path}")
            continue
        data = np.load(path)
        tracks = data["tracks"]
        indices = data["track_indices"] if "track_indices" in data.files else np.arange(tracks.shape[-1])

        if track_map:
            track_names = [track_map.get(int(idx), f"track_{idx}") for idx in indices]
            if args.right_labels:
                short = [shorten_track_label(name) for name in track_names]
                track_names = uniquify_labels(short, indices)

        y_label = args.y_label if args.y_label != "" else None
        fig = plot_tracks_waterfall(
            tracks,
            track_indices=list(indices),
            track_names=track_names,
            title=gene,
            normalize=args.normalize,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            y_scale=args.y_scale,
            y_label=y_label,
            right_labels=args.right_labels,
        )
        out_path = args.out_dir / f"{gene}{args.suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.clear()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
