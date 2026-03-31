"""Target track helpers for Enformer outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple
import urllib.request

import pandas as pd

TARGETS_URL = (
    "https://raw.githubusercontent.com/calico/basenji/master/manuscripts/"
    "cross2020/targets_human.txt"
)

DEFAULT_KEYWORDS = ("DNASE", "H3K27ac", "H3K4me3", "H3K4me1", "ATAC")


def ensure_targets_file(path: Path = Path("targets_human.txt"), url: str = TARGETS_URL) -> Path:
    """Download the human targets file if it does not exist."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_targets(path: Path) -> pd.DataFrame:
    """Load the Enformer target metadata table."""
    df = pd.read_csv(path, sep="\t")
    required = {"index", "description"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)} in {path}")
    return df


def select_regulatory_tracks(
    targets: pd.DataFrame,
    keywords: Sequence[str] = DEFAULT_KEYWORDS,
    max_tracks: Optional[int] = None,
) -> Tuple[list[int], list[str], pd.DataFrame]:
    """Select regulatory tracks by keyword match on the description."""
    if not keywords:
        raise ValueError("keywords must be a non-empty list of track filters")
    pattern = "|".join(keywords)
    mask = targets["description"].str.contains(pattern, case=False, na=False)
    selected = targets.loc[mask].copy()
    if max_tracks is not None:
        selected = selected.head(max_tracks)
    indices = selected["index"].astype(int).tolist()
    names = selected["description"].astype(str).tolist()
    return indices, names, selected


def write_track_selection(path: Path, track_indices: Iterable[int], track_names: Iterable[str]) -> Path:
    """Write track indices and names to CSV."""
    df = pd.DataFrame({"track_index": list(track_indices), "track_name": list(track_names)})
    df.to_csv(path, index=False)
    return path


def select_tracks_by_keyword_groups(
    targets: pd.DataFrame,
    keyword_groups: Mapping[str, Sequence[str]],
    limits: Optional[Mapping[str, int]] = None,
) -> Tuple[list[int], list[str], pd.DataFrame]:
    """Select unique tracks by grouped description keywords.

    Each entry in ``keyword_groups`` is matched against the target description.
    Duplicates across groups are removed while preserving first appearance.
    """
    selected_frames: list[pd.DataFrame] = []
    seen_indices: set[int] = set()

    for group_name, keywords in keyword_groups.items():
        if not keywords:
            continue
        pattern = "|".join(keywords)
        mask = targets["description"].str.contains(pattern, case=False, na=False, regex=True)
        group_df = targets.loc[mask].copy()
        if limits and group_name in limits:
            group_df = group_df.head(limits[group_name])
        if group_df.empty:
            continue
        group_df["keyword_group"] = group_name
        group_df = group_df[~group_df["index"].astype(int).isin(seen_indices)]
        seen_indices.update(group_df["index"].astype(int).tolist())
        selected_frames.append(group_df)

    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
    else:
        selected = targets.iloc[0:0].copy()
        selected["keyword_group"] = pd.Series(dtype=str)

    indices = selected["index"].astype(int).tolist()
    names = selected["description"].astype(str).tolist()
    return indices, names, selected
