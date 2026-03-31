from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import pyBigWig
except ImportError as exc:
    raise ImportError("pyBigWig is required. Install with `pip install pyBigWig`.") from exc

MUSCLE_TRACKS = [116, 117, 724, 737, 729, 741]
DEFAULT_GENE_PRIORITY = [
    "ACTA1", "ACTN2", "MYOD1", "MYH7", "TNNT1", "DES", "MYOG", "TTN", "MYLPF", "TPM2", "ADSS1"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute biologically matched TSS-local Pearson correlations for muscle-related tracks.")
    p.add_argument("--project-root", required=True)
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--targets-file", default=None)
    p.add_argument("--window-file", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--genes", default=None, help="Comma-separated gene list. If omitted, use gene list CSV if provided, else available NPZ genes.")
    p.add_argument("--gene-list", default=None, help="CSV file with official Gene Symbol column or gene column.")
    p.add_argument("--max-genes", type=int, default=None, help="Optional cap after filtering to genes with NPZ files.")
    p.add_argument("--tss-window-kb", type=float, default=4.0)
    p.add_argument("--track-indices", default=None, help="Optional comma-separated override for track indices.")
    return p.parse_args()


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


def relative_x_kb(n_bins: int, start: int, end: int, tss: int) -> np.ndarray:
    edges = np.linspace(start, end, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (centers - tss) / 1000.0


def fetch_bigwig_binned_signal(local_bigwig: Path, chrom: str, start: int, end: int, n_bins: int) -> np.ndarray:
    with pyBigWig.open(str(local_bigwig)) as bw:
        values = bw.stats(chrom, start, end, nBins=n_bins, type="mean")
    arr = np.array([0.0 if v is None or not np.isfinite(v) else float(v) for v in values], dtype=float)
    return arr


def compute_pearson_r(pred_signal: np.ndarray, exp_signal: np.ndarray) -> float:
    pred_signal = np.asarray(pred_signal, dtype=float)
    exp_signal = np.asarray(exp_signal, dtype=float)
    mask = np.isfinite(pred_signal) & np.isfinite(exp_signal)
    if mask.sum() < 3:
        return np.nan
    pred = pred_signal[mask]
    exp = exp_signal[mask]
    if np.std(pred) == 0 or np.std(exp) == 0:
        return np.nan
    return float(np.corrcoef(pred, exp)[0, 1])


def resolve_encode_bigwig_url(file_accession: str) -> tuple[str, str]:
    meta_url = f"https://www.encodeproject.org/files/{file_accession}/?format=json"
    headers = {"accept": "application/json"}
    r = requests.get(meta_url, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()
    href = data.get("href") or data.get("cloud_metadata", {}).get("url")
    if href is None:
        files = data.get("files") or []
        if files and isinstance(files, list):
            href = files[0].get("href")
    if href is None:
        raise RuntimeError(f"Could not resolve downloadable URL for {file_accession}")
    if href.startswith("http://") or href.startswith("https://"):
        download_url = href
    else:
        download_url = f"https://www.encodeproject.org{href}"
    encode_label = data.get("accession", file_accession)
    return download_url, encode_label


def download_bigwig_if_needed(accession: str, download_url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{accession}.bw"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with requests.get(download_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def load_gene_list_csv(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "official Gene Symbol" in fieldnames:
            col = "official Gene Symbol"
        elif "gene" in fieldnames:
            col = "gene"
        else:
            raise ValueError(f"No recognized gene column in {path}; found: {fieldnames}")
        genes = []
        for row in reader:
            g = (row.get(col) or "").strip()
            if g:
                genes.append(g)
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def build_gene_list(args: argparse.Namespace, outputs_dir: Path) -> tuple[list[str], list[str]]:
    available = sorted({p.stem.replace("_tracks", "") for p in outputs_dir.glob("*_tracks.npz")})
    available_set = set(available)

    if args.genes:
        requested = [g.strip() for g in args.genes.split(",") if g.strip()]
    elif args.gene_list:
        requested = load_gene_list_csv(Path(args.gene_list))
    else:
        requested = [g for g in DEFAULT_GENE_PRIORITY if g in available_set] + [g for g in available if g not in DEFAULT_GENE_PRIORITY]

    present = [g for g in requested if g in available_set]
    missing = [g for g in requested if g not in available_set]

    if args.max_genes is not None:
        present = present[: args.max_genes]

    return present, missing


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    outputs_dir = Path(args.outputs_dir)
    outdir = Path(args.outdir)
    targets_file = Path(args.targets_file) if args.targets_file else (project_root / "targets_human.txt")
    window_file = Path(args.window_file) if args.window_file else (outputs_dir / "gene_windows.csv")
    cache_dir = Path(args.cache_dir) if args.cache_dir else (project_root / "report" / "encode_bigwig_cache")

    outdir.mkdir(parents=True, exist_ok=True)

    if args.track_indices:
        track_indices = [int(x.strip()) for x in args.track_indices.split(",") if x.strip()]
    else:
        track_indices = MUSCLE_TRACKS

    genes, missing_genes = build_gene_list(args, outputs_dir)
    pd.DataFrame({"gene": genes}).to_csv(outdir / "selected_genes.csv", index=False)
    pd.DataFrame({"missing_gene": missing_genes}).to_csv(outdir / "missing_genes.csv", index=False)

    targets_df = load_targets_table(targets_file)
    windows_df = load_gene_windows(window_file)

    selected_df = targets_df[targets_df["index"].astype(int).isin(track_indices)].copy()
    selected_df["track_order"] = selected_df["index"].astype(int).map({idx: i for i, idx in enumerate(track_indices)})
    selected_df = selected_df.sort_values("track_order").reset_index(drop=True)
    selected_df.to_csv(outdir / "selected_biological_tracks.csv", index=False)

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

    metric_rows: list[dict] = []
    skipped: list[dict] = []

    for gene in genes:
        npz_path = outputs_dir / f"{gene}_tracks.npz"
        if not npz_path.exists():
            skipped.append({"gene": gene, "reason": "missing_npz"})
            continue

        row = windows_df.loc[windows_df["gene"] == gene]
        if row.empty:
            skipped.append({"gene": gene, "reason": "missing_gene_window"})
            continue
        row = row.iloc[0]

        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        tss = int(row["tss"])

        pred_matrix, present_track_indices = load_gene_npz(npz_path)
        col_of_track_index = {int(idx): i for i, idx in enumerate(present_track_indices)}
        x_kb = relative_x_kb(pred_matrix.shape[0], start, end, tss)

        local_mask = (x_kb >= -args.tss_window_kb) & (x_kb <= args.tss_window_kb)
        n_bins_used = int(local_mask.sum())
        if n_bins_used < 3:
            skipped.append({"gene": gene, "reason": "too_few_bins_in_window"})
            continue

        for idx in track_indices:
            if idx not in col_of_track_index:
                skipped.append({"gene": gene, "track_index": idx, "reason": "track_missing_in_npz"})
                continue

            meta = track_meta.get(idx)
            if meta is None:
                skipped.append({"gene": gene, "track_index": idx, "reason": "track_missing_in_targets"})
                continue

            pred_signal = pred_matrix[:, col_of_track_index[idx]].astype(float)
            try:
                exp_signal = fetch_bigwig_binned_signal(Path(meta["local_bigwig"]), chrom, start, end, pred_matrix.shape[0])
            except Exception as exc:
                skipped.append({"gene": gene, "track_index": idx, "reason": f"bigwig_read_failed: {exc}"})
                continue

            pearson_r = compute_pearson_r(pred_signal[local_mask], exp_signal[local_mask])
            metric_rows.append({
                "gene": gene,
                "track_index": idx,
                "track_desc": meta["description"],
                "encode_accession": meta["accession"],
                "encode_label": meta["encode_label"],
                "chrom": chrom,
                "start": start,
                "end": end,
                "tss": tss,
                "n_bins_full": int(pred_matrix.shape[0]),
                "n_bins_used": n_bins_used,
                "tss_window_kb": float(args.tss_window_kb),
                "pearson_r_tss_local": pearson_r,
                "x_kb_min": float(x_kb[local_mask].min()),
                "x_kb_max": float(x_kb[local_mask].max()),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "muscle_biological_tss_local_pearson_per_gene_track.csv", index=False)

    if not metrics_df.empty:
        by_track = (
            metrics_df.groupby(["track_index", "track_desc"], dropna=False)
            .agg(
                genes=("gene", "nunique"),
                mean_pearson_r=("pearson_r_tss_local", "mean"),
                median_pearson_r=("pearson_r_tss_local", "median"),
                min_pearson_r=("pearson_r_tss_local", "min"),
                max_pearson_r=("pearson_r_tss_local", "max"),
            )
            .reset_index()
        )
        by_track.to_csv(outdir / "muscle_biological_tss_local_pearson_summary_by_track.csv", index=False)

        overall = pd.DataFrame([
            {
                "genes_requested": len(genes) + len(missing_genes),
                "genes_with_npz": len(genes),
                "genes_missing_npz": len(missing_genes),
                "rows_evaluated": len(metrics_df),
                "mean_pearson_r": float(metrics_df["pearson_r_tss_local"].mean()),
                "median_pearson_r": float(metrics_df["pearson_r_tss_local"].median()),
            }
        ])
        overall.to_csv(outdir / "muscle_biological_tss_local_pearson_overall.csv", index=False)
    else:
        pd.DataFrame([{
            "genes_requested": len(genes) + len(missing_genes),
            "genes_with_npz": len(genes),
            "genes_missing_npz": len(missing_genes),
            "rows_evaluated": 0,
            "mean_pearson_r": np.nan,
            "median_pearson_r": np.nan,
        }]).to_csv(outdir / "muscle_biological_tss_local_pearson_overall.csv", index=False)

    pd.DataFrame(skipped).to_csv(outdir / "muscle_biological_tss_local_pearson_skipped.csv", index=False)
    print(json.dumps({
        "selected_genes": genes,
        "missing_genes": missing_genes,
        "track_indices": track_indices,
        "rows_evaluated": int(len(metrics_df)),
        "outdir": str(outdir),
    }, indent=2))


if __name__ == "__main__":
    main()
