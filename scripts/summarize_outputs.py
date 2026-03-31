#!/usr/bin/env python
"""Summarize Enformer batch outputs into CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.summary import write_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    gene_path, track_path = write_summaries(args.outputs)
    print(f"Wrote {gene_path}")
    print(f"Wrote {track_path}")


if __name__ == "__main__":
    main()
