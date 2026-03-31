#!/usr/bin/env python
"""Rank and shortlist track candidates from Lei's Pearson summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--min-genes", type=int, default=50)
    return parser.parse_args()


def to_float(value: str | None, default: float = float("-inf")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def main() -> None:
    args = parse_args()
    with args.summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [row for row in rows if to_int(row.get("genes")) >= args.min_genes]
    ranked = sorted(
        filtered,
        key=lambda row: (
            to_float(row.get("median_pearson_r")),
            to_float(row.get("mean_pearson_r")),
            to_float(row.get("min_pearson_r"), default=0.0),
        ),
        reverse=True,
    )

    selected = ranked[: args.top_n]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        fieldnames = [
            "track_index",
            "track_desc",
            "genes",
            "mean_pearson_r",
            "median_pearson_r",
            "min_pearson_r",
            "max_pearson_r",
            "rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            out_row = {name: row.get(name, "") for name in fieldnames}
            out_row["rank"] = rank
            writer.writerow(out_row)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
