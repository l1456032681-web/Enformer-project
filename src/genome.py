"""Genome helpers for hg38 sequence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import gzip
import shutil
from typing import Iterable, Optional, Tuple

HG38_FASTA_GZ = Path("hg38.fa.gz")
HG38_FASTA = Path("hg38.fa")


@dataclass(frozen=True)
class GeneWindow:
    gene: str
    chrom: str
    tss: int
    start: int
    end: int


def load_gene_symbols(csv_path: Path) -> list[str]:
    """Load gene symbols from the provided CSV list."""
    symbols: list[str] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "official Gene Symbol" not in reader.fieldnames:
            raise ValueError("Expected column 'official Gene Symbol' in gene list CSV")
        for row in reader:
            symbol = (row.get("official Gene Symbol") or "").strip()
            if symbol:
                symbols.append(symbol)
    return symbols


def ensure_uncompressed_fasta(
    fasta_gz_path: Path = HG38_FASTA_GZ,
    fasta_path: Optional[Path] = None,
) -> Path:
    """Ensure an uncompressed FASTA exists for random access.

    This streams the gzip file to disk. hg38 is large; expect several minutes
    and ~3 GB of disk usage.
    """
    if fasta_path is None:
        fasta_path = HG38_FASTA
    if fasta_path.exists():
        return fasta_path
    if not fasta_gz_path.exists():
        raise FileNotFoundError(f"Missing reference genome: {fasta_gz_path}")
    with gzip.open(fasta_gz_path, "rb") as src, fasta_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return fasta_path


def get_fasta_reader(fasta_path: Path):
    """Return a FASTA reader with random access support."""
    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise ImportError(
            "pyfaidx is required for FASTA random access. Install with 'pip install pyfaidx'."
        ) from exc
    return Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)


def normalize_chrom(chrom: str) -> str:
    chrom = chrom.strip()
    if not chrom:
        raise ValueError("Chromosome is empty")
    upper = chrom.upper()
    if upper in {"M", "MT", "CHRM", "CHRMT"}:
        return "chrM"
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return chrom


def centered_window(tss: int, window_size: int = 196_608) -> Tuple[int, int]:
    half = window_size // 2
    start = max(0, tss - half)
    end = start + window_size
    return start, end


def fetch_sequence(reader, chrom: str, start: int, end: int) -> str:
    """Fetch sequence in [start, end) from a FASTA reader."""
    chrom = normalize_chrom(chrom)
    seq = reader[chrom][start:end]
    return str(seq)


def pad_sequence(seq: str, expected_len: int, pad_char: str = "N") -> str:
    """Pad or trim sequence to expected length (right padding only)."""
    if len(seq) > expected_len:
        return seq[:expected_len]
    if len(seq) < expected_len:
        return seq + (pad_char * (expected_len - len(seq)))
    return seq


def validate_sequence_length(seq: str, expected_len: int) -> None:
    if len(seq) != expected_len:
        raise ValueError(f"Expected sequence length {expected_len}, got {len(seq)}")


def make_gene_window(gene: str, chrom: str, tss: int, window_size: int = 196_608) -> GeneWindow:
    start, end = centered_window(tss, window_size=window_size)
    return GeneWindow(gene=gene, chrom=normalize_chrom(chrom), tss=tss, start=start, end=end)
