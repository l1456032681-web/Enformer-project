"""ENCODE cCRE helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int
    label: str | None = None
    state: str | None = None


def load_ccre_bed(bed_path: Path) -> list[Interval]:
    intervals: list[Interval] = []
    with bed_path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[:3]
            label = parts[3] if len(parts) > 3 else None
            state = parts[5] if len(parts) > 5 else None
            intervals.append(
                Interval(chrom=chrom, start=int(start), end=int(end), label=label, state=state)
            )
    return intervals


def filter_intervals(intervals: Iterable[Interval], chrom: str, start: int, end: int) -> list[Interval]:
    out: list[Interval] = []
    for interval in intervals:
        if interval.chrom != chrom:
            continue
        if interval.end <= start or interval.start >= end:
            continue
        out.append(interval)
    return out


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def overlap_size(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return the number of overlapping bases between two half-open intervals."""
    if not intervals_overlap(a_start, a_end, b_start, b_end):
        return 0
    return min(a_end, b_end) - max(a_start, b_start)


def interval_distance(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return the distance between two intervals, or 0 if they overlap."""
    if intervals_overlap(a_start, a_end, b_start, b_end):
        return 0
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end


def overlapping_intervals(query: Interval, intervals: Iterable[Interval]) -> list[Interval]:
    """Return all intervals that overlap the query interval."""
    return [
        interval
        for interval in intervals
        if intervals_overlap(query.start, query.end, interval.start, interval.end)
    ]


def nearest_interval_distance(query: Interval, intervals: Iterable[Interval]) -> int | None:
    """Return the minimum distance from query to any interval, or None if no intervals exist."""
    distances = [
        interval_distance(query.start, query.end, interval.start, interval.end)
        for interval in intervals
    ]
    return min(distances) if distances else None


def compute_precision_recall(
    predicted: Iterable[Interval],
    truth: Iterable[Interval],
) -> tuple[float, float]:
    predicted = list(predicted)
    truth = list(truth)

    if not predicted:
        precision = 0.0
    else:
        hits = 0
        for pred in predicted:
            if any(intervals_overlap(pred.start, pred.end, t.start, t.end) for t in truth):
                hits += 1
        precision = hits / len(predicted)

    if not truth:
        recall = 0.0
    else:
        hits = 0
        for true in truth:
            if any(intervals_overlap(true.start, true.end, p.start, p.end) for p in predicted):
                hits += 1
        recall = hits / len(truth)

    return precision, recall
