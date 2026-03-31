from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    import pyBigWig
except ImportError as exc:
    raise ImportError("pyBigWig is required. Install with `pip install pyBigWig`.") from exc

PRED_COLOR = "#3A78C2"
EXP_COLOR = "#56B881"
GRID_COLOR = "#EAEAEA"
TEXT_COLOR = "#333333"
ENFORMER_BIN_SIZE = 128


def load_targets_table(targets_path: Path) -> pd.DataFrame:
    df = pd.read_csv(targets_path, sep="\t")
    required = {"index", "identifier", "description"}
    if not required.issubset(df.columns):
        raise ValueError(f"targets file missing required columns: {targets_path}")
    df["index"] = df["index"].astype(int)
    return df


def load_gene_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"gene", "chrom", "start", "end", "tss"}
    if not required.issubset(df.columns):
        raise ValueError(f"gene_windows.csv missing required columns: {path}")
    return df


def load_gene_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    tracks = np.asarray(data["tracks"])
    track_indices = np.asarray(data["track_indices"]).astype(int)
    if tracks.ndim != 3:
        raise ValueError(f"Unexpected tracks shape in {npz_path}: {tracks.shape}")
    return tracks[0], track_indices


def load_gene_list(path: Path) -> list[str]:
    df = pd.read_csv(path)
    for col in ["official Gene Symbol", "gene", "Gene", "symbol"]:
        if col in df.columns:
            genes = [str(x).strip() for x in df[col].tolist() if str(x).strip() and str(x) != "nan"]
            seen: set[str] = set()
            out: list[str] = []
            for g in genes:
                if g not in seen:
                    seen.add(g)
                    out.append(g)
            return out
    raise ValueError(f"Could not find a gene-symbol column in {path}. Columns: {list(df.columns)}")


def parse_pair_csv(pair_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pair_csv)
    required = {"gene", "track_index"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_csv} must contain columns: {sorted(required)}")
    df["gene"] = df["gene"].astype(str)
    df["track_index"] = df["track_index"].astype(int)
    if "display_order" not in df.columns:
        df["display_order"] = np.arange(len(df))
    return df.sort_values(["gene", "display_order"]).reset_index(drop=True)


def build_pair_table(
    pair_csv: Path | None,
    genes_arg: str | None,
    gene_list_file: Path | None,
    track_indices_arg: str | None,
    outputs_dir: Path,
) -> pd.DataFrame:
    if pair_csv is not None:
        return parse_pair_csv(pair_csv)

    if not track_indices_arg:
        raise ValueError("Provide either --pair-csv or --track-indices (optionally with --genes or --gene-list)")

    if genes_arg:
        genes = [g.strip() for g in genes_arg.split(",") if g.strip()]
    elif gene_list_file is not None:
        genes = load_gene_list(gene_list_file)
    else:
        genes = []

    if not genes:
        raise ValueError("No genes were provided. Use --genes or --gene-list.")

    available = {p.name.replace("_tracks.npz", "") for p in outputs_dir.glob("*_tracks.npz")}
    genes = [g for g in genes if g in available]
    track_indices = [int(x.strip()) for x in track_indices_arg.split(",") if x.strip()]

    rows = []
    for gene in genes:
        for order, track_idx in enumerate(track_indices):
            rows.append({"gene": gene, "track_index": track_idx, "display_order": order})
    return pd.DataFrame(rows)


def resolve_encode_bigwig_url(file_accession: str) -> tuple[str, str]:
    meta_url = f"https://www.encodeproject.org/files/{file_accession}/?format=json"
    headers = {"accept": "application/json"}
    r = requests.get(meta_url, headers=headers, timeout=120)
    r.raise_for_status()
    info = r.json()

    href = info.get("href")
    title = str(info.get("title") or file_accession)
    file_format = str(info.get("file_format", "")).lower()

    if file_format != "bigwig":
        raise RuntimeError(f"{file_accession} is not a bigWig file (file_format={file_format})")
    if not href:
        raise RuntimeError(f"{file_accession} metadata has no href")

    download_url = "https://www.encodeproject.org" + href if str(href).startswith("/") else str(href)
    return download_url, title


def download_bigwig_if_needed(file_accession: str, download_url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{file_accession}.bigWig"

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    print(f"Downloading {file_accession} -> {local_path}")
    with requests.get(download_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return local_path


def infer_prediction_span(start: int, end: int, n_bins: int, bin_size: int = ENFORMER_BIN_SIZE) -> tuple[int, int]:
    """
    Align Enformer predictions to the central output crop.
    For 196,608 bp input and 896 bins at 128 bp, the predicted span is 114,688 bp
    centered within the full input window, leaving a 40,960 bp margin on each side.
    """
    window_size = int(end) - int(start)
    pred_span = int(n_bins) * int(bin_size)
    if pred_span > window_size:
        raise ValueError(f"Prediction span ({pred_span}) exceeds window size ({window_size})")
    left_crop = (window_size - pred_span) // 2
    pred_start = int(start) + left_crop
    pred_end = pred_start + pred_span
    return pred_start, pred_end


def prediction_bin_centers_rel_kb(n_bins: int, start: int, end: int, tss: int, bin_size: int = ENFORMER_BIN_SIZE) -> np.ndarray:
    pred_start, _ = infer_prediction_span(start, end, n_bins, bin_size=bin_size)
    centers = pred_start + (np.arange(n_bins, dtype=float) + 0.5) * float(bin_size)
    return (centers - float(tss)) / 1000.0


def fetch_bigwig_signal_aligned_to_prediction(local_bigwig_path: Path, chrom: str, start: int, end: int, n_bins: int, bin_size: int = ENFORMER_BIN_SIZE) -> np.ndarray:
    pred_start, pred_end = infer_prediction_span(start, end, n_bins, bin_size=bin_size)
    bw = pyBigWig.open(str(local_bigwig_path))
    if bw is None:
        raise RuntimeError(f"Could not open local bigWig: {local_bigwig_path}")
    try:
        vals = bw.stats(chrom, pred_start, pred_end, nBins=n_bins, type="mean")
    finally:
        bw.close()

    return np.array(
        [0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v) for v in vals],
        dtype=float,
    )


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    lo = np.nanmin(x[good])
    hi = np.nanmax(x[good])
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_prediction_only(gene: str, chrom: str, x_kb: np.ndarray, rows: list[dict], out_path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(12.8, max(4.8, 2.3 * n)), sharex=True, gridspec_kw={"hspace": 0.08})
    if n == 1:
        axes = [axes]

    fig.suptitle(gene, fontsize=16, y=0.995)
    fig.text(0.05, 0.972, "Prediction", color=PRED_COLOR, fontsize=11, weight="bold")

    for i, row in enumerate(rows):
        ax = axes[i]
        y = normalize_for_display(row["pred_signal"])
        ax.fill_between(x_kb, 0, y, color=PRED_COLOR, alpha=0.95, linewidth=0)
        ax.plot(x_kb, y, color=PRED_COLOR, linewidth=0.6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=9)

        ax.text(1.006, 0.5, row["track_desc"], transform=ax.transAxes,
                ha="left", va="center", fontsize=10, color=TEXT_COLOR)

    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel(f"Position relative to TSS (kb)   |   {chrom}", fontsize=11)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_exp(gene: str, chrom: str, x_kb: np.ndarray, rows: list[dict], out_path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(2 * n, 1, figsize=(12.8, max(7.2, 1.5 * 2 * n + 1.2)),
                             sharex=True, gridspec_kw={"hspace": 0.08})
    if 2 * n == 1:
        axes = [axes]

    fig.suptitle(gene, fontsize=16, y=0.995)
    fig.text(0.05, 0.972, "Prediction", color=PRED_COLOR, fontsize=11, weight="bold")
    fig.text(0.05, 0.946, "Experiment", color=EXP_COLOR, fontsize=11, weight="bold")

    for i, row in enumerate(rows):
        ax_pred = axes[2 * i]
        ax_exp = axes[2 * i + 1]

        y_pred = normalize_for_display(row["pred_signal"])
        y_exp = normalize_for_display(row["exp_signal"])

        ax_pred.fill_between(x_kb, 0, y_pred, color=PRED_COLOR, alpha=0.95, linewidth=0)
        ax_pred.plot(x_kb, y_pred, color=PRED_COLOR, linewidth=0.6)

        ax_exp.fill_between(x_kb, 0, y_exp, color=EXP_COLOR, alpha=0.95, linewidth=0)
        ax_exp.plot(x_kb, y_exp, color=EXP_COLOR, linewidth=0.6)

        for ax in (ax_pred, ax_exp):
            ax.set_ylim(0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_color("#999999")
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=9)

        ax_pred.text(1.006, 0.52, row["track_desc"], transform=ax_pred.transAxes,
                     ha="left", va="center", fontsize=10, color=TEXT_COLOR)
        ax_exp.text(1.006, 0.25, row["encode_label"], transform=ax_exp.transAxes,
                    ha="left", va="center", fontsize=8.5, color="#666666")

    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel(f"Position relative to TSS (kb)   |   {chrom}", fontsize=11)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--targets-file", default=None)
    p.add_argument("--window-file", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--pair-csv", default=None)
    p.add_argument("--genes", default=None)
    p.add_argument("--gene-list", default=None)
    p.add_argument("--track-indices", default=None)
    args = p.parse_args()

    project_root = Path(args.project_root)
    outputs_dir = Path(args.outputs_dir)
    outdir = Path(args.outdir)
    targets_file = Path(args.targets_file) if args.targets_file else (project_root / "targets_human.txt")
    window_file = Path(args.window_file) if args.window_file else (outputs_dir / "gene_windows.csv")
    cache_dir = Path(args.cache_dir) if args.cache_dir else (project_root / "report" / "encode_bigwig_cache")
    pair_csv = Path(args.pair_csv) if args.pair_csv else None
    gene_list_file = Path(args.gene_list) if args.gene_list else None

    pair_df = build_pair_table(pair_csv, args.genes, gene_list_file, args.track_indices, outputs_dir)
    if pair_df.empty:
        raise ValueError("No usable gene-track pairs were found. Check gene list, outputs-dir, and track indices.")

    targets_df = load_targets_table(targets_file)
    windows_df = load_gene_windows(window_file)

    unique_tracks = sorted(pair_df["track_index"].astype(int).unique().tolist())
    selected_df = targets_df[targets_df["index"].isin(unique_tracks)].copy()
    if selected_df["index"].nunique() != len(unique_tracks):
        missing = sorted(set(unique_tracks) - set(selected_df["index"].tolist()))
        raise ValueError(f"Track indices not found in targets file: {missing}")

    track_meta = {}
    for _, row in selected_df.iterrows():
        idx = int(row["index"])
        acc = str(row["identifier"])
        desc = str(row["description"])
        download_url, encode_label = resolve_encode_bigwig_url(acc)
        local_bigwig = download_bigwig_if_needed(acc, download_url, cache_dir)
        track_meta[idx] = {
            "accession": acc,
            "description": desc,
            "download_url": download_url,
            "local_bigwig": str(local_bigwig),
            "encode_label": encode_label,
        }

    print("Resolved ENCODE files:")
    print(json.dumps(track_meta, indent=2))

    for gene, group in pair_df.groupby("gene", sort=False):
        npz_path = outputs_dir / f"{gene}_tracks.npz"
        if not npz_path.exists():
            print(f"[skip] missing prediction file: {npz_path}")
            continue

        row = windows_df.loc[windows_df["gene"] == gene]
        if row.empty:
            print(f"[skip] {gene} not found in gene_windows.csv")
            continue
        row = row.iloc[0]

        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        tss = int(row["tss"])

        pred_matrix, track_indices = load_gene_npz(npz_path)
        col_of_track_index = {int(idx): i for i, idx in enumerate(track_indices)}
        x_kb = prediction_bin_centers_rel_kb(pred_matrix.shape[0], start, end, tss)

        rows = []
        for _, pair_row in group.sort_values("display_order").iterrows():
            idx = int(pair_row["track_index"])
            if idx not in col_of_track_index:
                print(f"[skip] {gene}: track {idx} not present in NPZ")
                continue

            meta = track_meta[idx]
            pred_signal = pred_matrix[:, col_of_track_index[idx]]

            try:
                exp_signal = fetch_bigwig_signal_aligned_to_prediction(
                    Path(meta["local_bigwig"]), chrom, start, end, pred_matrix.shape[0]
                )
            except Exception as exc:
                print(f"[skip] {gene}: failed to read local experiment for track {idx} ({meta['accession']}) -> {exc}")
                continue

            rows.append({
                "track_index": idx,
                "track_desc": meta["description"],
                "encode_label": meta["accession"],
                "pred_signal": pred_signal,
                "exp_signal": exp_signal,
            })

        if not rows:
            print(f"[skip] {gene}: no usable rows")
            continue

        pred_only_path = outdir / f"{gene}_pred_only_matched_layout.png"
        pred_vs_exp_path = outdir / f"{gene}_pred_vs_exp_matched_layout.png"

        plot_prediction_only(gene, chrom, x_kb, rows, pred_only_path)
        plot_pred_vs_exp(gene, chrom, x_kb, rows, pred_vs_exp_path)

        print(f"[saved] {pred_only_path}")
        print(f"[saved] {pred_vs_exp_path}")


if __name__ == "__main__":
    main()
