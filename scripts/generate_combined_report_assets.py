#!/usr/bin/env python
"""Generate figures and appendix tables for the combined Enformer report."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from src.encode import filter_intervals, load_ccre_bed
from src.plotting import plot_locus_overlay

SHOWCASE_GENES = ["ACTA1", "ACTN2", "ADSS1", "MYOD1"]

OLD_OVERLAP_CSV = textwrap.dedent(
    """\
    gene,precision,recall,predicted_peaks,ccre_in_window,f1
    ASCC3,1.000,0.044,3,45,0.085
    ACTN2,0.500,0.013,6,76,0.026
    ASCC1,0.333,0.011,6,179,0.022
    ACTA1,1.000,0.010,1,99,0.020
    ADSS1,1.000,0.004,1,247,0.008
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccre-bed", type=Path, required=True)
    parser.add_argument("--old-outputs", type=Path, default=Path("enformer_results/outputs"))
    parser.add_argument(
        "--muscle-outputs",
        type=Path,
        default=Path("report/muscle_bundle_v2/outputs_muscle"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("report_combined"))
    return parser.parse_args()


def choose_main_track(track_selection: pd.DataFrame) -> tuple[int, str]:
    cage = track_selection[track_selection["track_name"].str.contains("CAGE", case=False, na=False)]
    if not cage.empty:
        row = cage.iloc[0]
        return int(row["track_index"]), str(row["track_name"])

    h3k27ac = track_selection[
        track_selection["track_name"].str.contains("H3K27ac", case=False, na=False)
    ]
    if not h3k27ac.empty:
        row = h3k27ac.iloc[0]
        return int(row["track_index"]), str(row["track_name"])

    row = track_selection.iloc[0]
    return int(row["track_index"]), str(row["track_name"])


def track_values_from_npz(npz_path: Path, global_track_index: int) -> np.ndarray:
    data = np.load(npz_path)
    track_indices = data["track_indices"]
    local_matches = np.where(track_indices == global_track_index)[0]
    if len(local_matches) == 0:
        raise ValueError(f"Track index {global_track_index} not found in {npz_path}")
    local_idx = int(local_matches[0])
    return data["tracks"][0, :, local_idx]


def load_predicted_peaks(peaks_path: Path) -> list[tuple[int, int]]:
    peaks: list[tuple[int, int]] = []
    if not peaks_path.exists():
        return peaks
    with peaks_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            peaks.append((int(parts[1]), int(parts[2])))
    return peaks


def build_gene_summary_table(gene_summary_path: Path, out_path: Path) -> None:
    df = pd.read_csv(gene_summary_path)
    df = df.sort_values(["gene", "max_signal"], ascending=[True, False])
    df = df.groupby("gene", as_index=False).first()
    df = df[["gene", "track_name", "max_signal", "center_signal"]].copy()
    df = df.rename(
        columns={
            "track_name": "track name",
            "max_signal": "max signal",
            "center_signal": "center signal",
        }
    )
    df["track name"] = df["track name"].str.slice(0, 42)
    latex = df.to_latex(
        index=False,
        longtable=True,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x),
        caption="Summary of all processed genes from the original assay-balanced run.",
        label="tab:all_gene_summary",
    )
    out_path.write_text(latex)


def build_overview_table(old_outputs: Path, muscle_outputs: Path, out_path: Path) -> None:
    old_tracks = pd.read_csv(old_outputs / "track_selection.csv")
    old_overlap = pd.read_csv(StringIO(OLD_OVERLAP_CSV))
    muscle_tracks = pd.read_csv(muscle_outputs / "track_selection.csv")
    muscle_overlap = pd.read_csv(muscle_outputs / "ccre_overlap_with_f1.csv")

    rows = [
        {
            "run": "Original assay-balanced",
            "tracks": len(old_tracks),
            "genes_processed": len(list(old_outputs.glob("*_tracks.npz"))),
            "genes_with_overlap": len(old_overlap),
            "mean_precision": old_overlap["precision"].mean(),
            "mean_recall": old_overlap["recall"].mean(),
        },
        {
            "run": "Muscle-focused",
            "tracks": len(muscle_tracks),
            "genes_processed": len(list(muscle_outputs.glob("*_tracks.npz"))),
            "genes_with_overlap": len(muscle_overlap),
            "mean_precision": muscle_overlap["precision"].mean(),
            "mean_recall": muscle_overlap["recall"].mean(),
        },
    ]
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "genes_processed": "genes processed",
            "genes_with_overlap": "genes with overlap",
            "mean_precision": "mean precision",
            "mean_recall": "mean recall",
        }
    )
    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Overview of the original and muscle-focused analyses.",
        label="tab:overview",
    )
    out_path.write_text(latex)


def build_muscle_track_table(detailed_path: Path, out_path: Path) -> None:
    df = pd.read_csv(detailed_path)
    df = df[["index", "keyword_group", "description"]].copy()
    df = df.rename(columns={"keyword_group": "keyword group"})
    df["description"] = df["description"].str.slice(0, 75)
    latex = df.to_latex(
        index=False,
        longtable=True,
        caption="Muscle-focused Enformer target selection.",
        label="tab:muscle_tracks",
    )
    out_path.write_text(latex)


def build_processed_genes_table(gene_windows_path: Path, out_path: Path) -> None:
    df = pd.read_csv(gene_windows_path)
    genes = sorted(df["gene"].dropna().astype(str).tolist())
    rows = []
    for i in range(0, len(genes), 3):
        row = genes[i : i + 3]
        while len(row) < 3:
            row.append("")
        rows.append(row)
    out_df = pd.DataFrame(rows, columns=["Gene 1", "Gene 2", "Gene 3"])
    latex = out_df.to_latex(
        index=False,
        longtable=True,
        caption="All processed genes from the original project workflow.",
        label="tab:processed_genes",
    )
    out_path.write_text(latex)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ccre = load_ccre_bed(args.ccre_bed)
    old_windows = pd.read_csv(args.old_outputs / "gene_windows.csv")
    muscle_windows = pd.read_csv(args.muscle_outputs / "gene_windows.csv")
    track_selection = pd.read_csv(args.muscle_outputs / "track_selection.csv")
    main_track_index, main_track_label = choose_main_track(track_selection)

    for gene in SHOWCASE_GENES:
        row = muscle_windows[muscle_windows["gene"] == gene]
        if row.empty:
            continue
        row = row.iloc[0]
        window_start = int(row["start"])
        window_end = int(row["end"])
        chrom = str(row["chrom"])

        track_values = track_values_from_npz(args.muscle_outputs / f"{gene}_tracks.npz", main_track_index)

        attr_npz = np.load(args.muscle_outputs / "attribution" / f"{gene}_attr.npz")
        attribution = attr_npz["attribution"]
        predicted_peaks = load_predicted_peaks(args.muscle_outputs / "attribution" / f"{gene}_peaks.bed")
        nearby = old_windows[
            (old_windows["chrom"] == chrom)
            & (old_windows["tss"] >= window_start)
            & (old_windows["tss"] <= window_end)
        ][["gene", "tss"]].to_dict(orient="records")
        ccre_intervals = filter_intervals(ccre, chrom, window_start, window_end)

        fig = plot_locus_overlay(
            gene=gene,
            chrom=chrom,
            window_start=window_start,
            window_end=window_end,
            track_values=track_values,
            track_label=main_track_label,
            attribution=attribution,
            ccre_intervals=ccre_intervals,
            nearby_tss=nearby,
            predicted_peaks=predicted_peaks,
        )
        fig.savefig(figures_dir / f"overlay_{gene}.png", dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

    build_overview_table(args.old_outputs, args.muscle_outputs, args.report_dir / "overview_table.tex")
    build_muscle_track_table(
        args.muscle_outputs / "track_selection_muscle_detailed.csv",
        args.report_dir / "muscle_track_table.tex",
    )
    build_gene_summary_table(
        args.old_outputs / "gene_track_summary.csv",
        args.report_dir / "all_gene_summary_table.tex",
    )
    build_processed_genes_table(
        args.old_outputs / "gene_windows.csv",
        args.report_dir / "processed_genes_table.tex",
    )


if __name__ == "__main__":
    main()
