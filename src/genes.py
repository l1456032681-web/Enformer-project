"""Gene coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable, Optional


@dataclass(frozen=True)
class GeneCoord:
    gene: str
    chrom: str
    tss: int
    strand: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


def coords_to_dataframe(coords: Iterable[GeneCoord]):
    """Convert gene coordinates to a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "gene": coord.gene,
                "chrom": coord.chrom,
                "tss": coord.tss,
                "strand": coord.strand,
                "start": coord.start,
                "end": coord.end,
            }
            for coord in coords
        ]
    )


def write_gene_coordinates(coords: Iterable[GeneCoord], csv_path: Path) -> Path:
    """Write gene coordinates to CSV for reuse."""
    df = coords_to_dataframe(coords)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_gene_coordinates(csv_path: Path) -> list[GeneCoord]:
    """Load gene coordinates from a CSV with gene/chrom/tss columns.

    Expected columns (case-insensitive): gene, chrom, tss.
    TSS is 0-based coordinate for Python slicing.
    """
    from src.genome import normalize_chrom

    coords: list[GeneCoord] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.lower() for name in (reader.fieldnames or [])]
        if not {"gene", "chrom", "tss"}.issubset(fieldnames):
            raise ValueError("Expected columns: gene, chrom, tss")
        for row in reader:
            gene = (row.get("gene") or row.get("Gene") or "").strip()
            chrom = (row.get("chrom") or row.get("Chrom") or "").strip()
            tss = row.get("tss") or row.get("TSS")
            if not gene or not chrom or tss is None:
                continue
            coords.append(GeneCoord(gene=gene, chrom=normalize_chrom(chrom), tss=int(tss)))
    return coords


def fetch_gene_coordinates_mygene(symbols: Iterable[str]) -> list[GeneCoord]:
    """Fetch hg38 coordinates for gene symbols using mygene.info.

    This uses the public API; coordinates are converted to 0-based for slicing.
    """
    import requests
    import warnings
    from src.genome import normalize_chrom

    def normalize_entries(pos) -> list[dict]:
        if isinstance(pos, dict):
            candidates = [pos]
        elif isinstance(pos, list):
            candidates = [p for p in pos if isinstance(p, dict)]
        else:
            return []

        entries: list[dict] = []
        for entry in candidates:
            chrom = entry.get("chr")
            start = entry.get("start")
            end = entry.get("end")
            if chrom is None or start is None or end is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            entries.append(
                {
                    "chrom": str(chrom),
                    "start": start_i,
                    "end": end_i,
                    "strand": entry.get("strand"),
                }
            )
        return entries

    def is_canonical(chrom: str) -> bool:
        chrom = chrom.replace("chr", "")
        return chrom.isdigit() or chrom in {"X", "Y", "M", "MT"}

    def pick_gene_coords(entries: list[dict]):
        if not entries:
            return None
        candidates = [e for e in entries if is_canonical(e["chrom"])] or entries
        by_chrom: dict[str, list[dict]] = {}
        for entry in candidates:
            by_chrom.setdefault(entry["chrom"], []).append(entry)
        chrom = max(by_chrom, key=lambda c: len(by_chrom[c]))
        selected = by_chrom[chrom]

        strand = next((e["strand"] for e in selected if e.get("strand") is not None), None)
        start = min(e["start"] for e in selected)
        end = max(e["end"] for e in selected)

        if strand in (-1, "-1", "-"):
            tss = end - 1
            strand_symbol = "-"
        else:
            tss = start - 1
            strand_symbol = "+"

        return chrom, start, end, tss, strand_symbol

    coords: list[GeneCoord] = []
    fallback_genes: list[str] = []
    for symbol in symbols:
        params = {
            "q": symbol,
            "species": "human",
            "fields": "symbol,genomic_pos_hg38,genomic_pos",
            "size": 1,
        }
        response = requests.get("https://mygene.info/v3/query", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", [])
        if not hits:
            continue
        hit = hits[0]
        pos = hit.get("genomic_pos_hg38")
        if not pos:
            pos = hit.get("genomic_pos")
            if pos:
                fallback_genes.append(symbol)
        if not pos:
            continue
        entries = normalize_entries(pos)
        picked = pick_gene_coords(entries)
        if not picked:
            continue
        chrom, start, end, tss, strand_symbol = picked
        chrom = normalize_chrom(chrom)
        coords.append(
            GeneCoord(
                gene=symbol,
                chrom=chrom,
                tss=tss,
                strand=strand_symbol,
                start=start - 1,
                end=end,
            )
        )

    if fallback_genes:
        sample = ", ".join(fallback_genes[:5])
        warnings.warn(
            "Used genomic_pos for {} genes (no genomic_pos_hg38). Example: {}".format(
                len(fallback_genes), sample
            ),
            RuntimeWarning,
        )

    return coords
