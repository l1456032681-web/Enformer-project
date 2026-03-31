from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.attribution import (
    attribution_peaks_to_intervals,
    enformer_abs_attribution_profile,
    enformer_bin_to_genome_interval,
    enformer_bins_to_genome_coords,
    gradient_x_input_for_track_bins,
)
from src.encode import (
    compute_f1,
    compute_precision_recall,
    interval_jaccard,
    load_ccre_bed,
    restrict_ccre_truth_set,
)
from src.enformer import load_enformer, predict_tracks, prepare_sequence_tensor
from src.genes import fetch_gene_coordinates_mygene, load_gene_coordinates, write_gene_coordinates
from src.genome import (
    ensure_uncompressed_fasta,
    fetch_sequence,
    get_fasta_reader,
    load_gene_symbols,
    make_gene_window,
    pad_sequence,
    validate_sequence_length,
)
from src.plotting import plot_four_gene_overview, plot_gene_overlay
from src.targets import choose_best_muscle_regulatory_track, load_targets, rank_muscle_regulatory_tracks


DEFAULT_FIGURE_GENES = ["ACTA1", "MYH7", "TNNT1", "DES"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 4-gene overlay figures and metrics.")
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--track-index",
        type=int,
        default=None,
        help="Optional fixed Enformer track index. If omitted, a muscle-regulatory track is chosen automatically.",
    )
    parser.add_argument(
        "--genes",
        default=",".join(DEFAULT_FIGURE_GENES),
        help="Exactly 4 comma-separated genes to visualize.",
    )
    parser.add_argument("--ccre-bed", required=True)
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--coords-csv",
        default=None,
        help="Optional cached hg38 gene coordinates CSV with columns gene/chrom/tss.",
    )
    parser.add_argument("--fasta-gz", default="/content/drive/MyDrive/enformer_data/hg38.fa.gz")
    parser.add_argument("--fasta-out", default="hg38.fa")
    parser.add_argument("--prediction-device", default="cuda")
    parser.add_argument("--attribution-device", default="cpu")
    parser.add_argument("--peak-quantile", type=float, default=0.995)
    parser.add_argument("--peak-min-width", type=int, default=64)
    parser.add_argument("--smooth-window", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=196608)
    parser.add_argument("--tss-flank-bp", type=int, default=20000)
    parser.add_argument("--anchor-flank-bp", type=int, default=10000)
    parser.add_argument("--include-ctcf", action="store_true", default=False)
    return parser.parse_args()


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device_name: str) -> str:
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def load_or_fetch_coords(genes: list[str], coords_csv: str | None) -> dict[str, object]:
    coord_map = {}
    missing = list(genes)

    if coords_csv:
        coords_path = Path(coords_csv)
        if coords_path.exists():
            cached = load_gene_coordinates(coords_path)
            coord_map.update({c.gene: c for c in cached})
            missing = [g for g in genes if g not in coord_map]

    if missing:
        fetched = fetch_gene_coordinates_mygene(missing)
        coord_map.update({c.gene: c for c in fetched})
        if coords_csv:
            write_gene_coordinates(coord_map.values(), Path(coords_csv))

    return coord_map


def main() -> None:
    args = parse_args()

    output_dir = Path(args.outputs_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    figure_genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    if len(figure_genes) != 4:
        raise ValueError("--genes must contain exactly 4 comma-separated genes")

    prediction_device = resolve_device(args.prediction_device)
    attribution_device = resolve_device(args.attribution_device)
    print("Prediction device:", prediction_device)
    print("Attribution device:", attribution_device)

    targets = load_targets(Path(args.targets))
    ranked_tracks = rank_muscle_regulatory_tracks(targets, top_n=50)
    ranked_tracks.to_csv(output_dir / "candidate_muscle_tracks.csv", index=False)

    if args.track_index is None:
        track_index, track_name, _ = choose_best_muscle_regulatory_track(targets)
        print(f"Auto-selected track {track_index}: {track_name}")
    else:
        track_index = int(args.track_index)
        row = targets.loc[targets["index"].astype(int) == track_index, "description"]
        if row.empty:
            raise ValueError(f"Track index {track_index} not found in {args.targets}")
        track_name = str(row.iloc[0])

    pd.DataFrame(
        [{"track_index": track_index, "track_name": track_name}]
    ).to_csv(output_dir / "track_selection.csv", index=False)

    all_genes_in_list = load_gene_symbols(Path(args.gene_list))
    gene_set = set(all_genes_in_list)
    missing_fig = [g for g in figure_genes if g not in gene_set]
    if missing_fig:
        raise ValueError(f"These genes are not in {args.gene_list}: {missing_fig}")

    coord_map = load_or_fetch_coords(figure_genes, args.coords_csv)

    fasta_path = ensure_uncompressed_fasta(Path(args.fasta_gz), Path(args.fasta_out))
    fasta = get_fasta_reader(fasta_path)
    ccre_intervals = load_ccre_bed(Path(args.ccre_bed))

    metric_rows: list[dict] = []
    four_gene_panels: list[dict] = []

    for i, gene in enumerate(figure_genes, start=1):
        print(f"=== Processing {gene} ({i}/4) ===")
        coord = coord_map.get(gene)
        if coord is None:
            print(f"Skipping {gene}: no coordinates found")
            continue

        window = make_gene_window(gene, coord.chrom, coord.tss, window_size=args.window_size)
        seq = fetch_sequence(fasta, window.chrom, window.start, window.end)
        seq = pad_sequence(seq, args.window_size)
        validate_sequence_length(seq, args.window_size)

        pred_model = load_enformer(device=prediction_device)
        pred_tokens = prepare_sequence_tensor(seq, device=prediction_device)
        preds = predict_tracks(pred_model, pred_tokens)
        preds_human = preds["human"] if isinstance(preds, dict) else preds

        pred_signal = preds_human[0, :, track_index].detach().cpu().numpy()
        bin_starts = enformer_bins_to_genome_coords(len(pred_signal), window.start)
        best_bin = int(np.argmax(pred_signal))
        best_bin_start, best_bin_end = enformer_bin_to_genome_interval(best_bin, window.start)

        # Use the strongest-response bin as the attribution target anchor.
        # This keeps the original workflow intuition while still using
        # absolute gradient x input followed by avg-pool style smoothing.
        target_bins = sorted(
            {
                max(0, best_bin - 1),
                best_bin,
                min(len(bin_starts) - 1, best_bin + 1),
            }
        )

        del preds, preds_human, pred_tokens, pred_model
        cleanup_memory()

        attr_model = load_enformer(device=attribution_device)
        attr_tokens = prepare_sequence_tensor(seq, device=attribution_device)
        attr = gradient_x_input_for_track_bins(
            model=attr_model,
            tokens=attr_tokens,
            track_index=track_index,
            bin_indices=target_bins,
            head="human",
        )
        attr_scores = enformer_abs_attribution_profile(attr[0], window=args.smooth_window)

        del attr, attr_tokens, attr_model
        cleanup_memory()

        predicted_intervals = attribution_peaks_to_intervals(
            chrom=window.chrom,
            window_start=window.start,
            scores=attr_scores,
            threshold_quantile=args.peak_quantile,
            min_width=args.peak_min_width,
        )
        candidate_intervals = restrict_ccre_truth_set(
            ccre_intervals,
            window.chrom,
            window.start,
            window.end,
            tss=coord.tss,
            anchor_start=best_bin_start,
            anchor_end=best_bin_end,
            tss_flank_bp=args.tss_flank_bp,
            anchor_flank_bp=args.anchor_flank_bp,
            include_ctcf=args.include_ctcf,
        )

        peak_ccre_precision, peak_ccre_recall = compute_precision_recall(
            predicted_intervals,
            candidate_intervals,
        )
        peak_ccre_f1 = compute_f1(peak_ccre_precision, peak_ccre_recall)
        peak_ccre_jaccard_bp = interval_jaccard(predicted_intervals, candidate_intervals)

        metric_rows.append(
            {
                "gene": gene,
                "chrom": window.chrom,
                "tss": coord.tss,
                "window_start": window.start,
                "window_end": window.end,
                "track_index": track_index,
                "track_name": track_name,
                "best_bin": best_bin,
                "target_bins": ",".join(map(str, target_bins)),
                "best_bin_genome_start": int(bin_starts[best_bin]),
                "best_bin_genome_end": int(bin_starts[best_bin] + 128),
                "n_predicted_peaks": len(predicted_intervals),
                "n_candidate_ccres": len(candidate_intervals),
                "peak_ccre_precision": peak_ccre_precision,
                "peak_ccre_recall": peak_ccre_recall,
                "peak_ccre_f1": peak_ccre_f1,
                "peak_ccre_jaccard_bp": peak_ccre_jaccard_bp,
            }
        )

        fig = plot_gene_overlay(
            gene=gene,
            chrom=window.chrom,
            window_start=window.start,
            tss=coord.tss,
            bin_starts=bin_starts,
            pred_signal=pred_signal,
            attribution_scores=attr_scores,
            candidate_intervals=candidate_intervals,
            predicted_intervals=predicted_intervals,
            track_name=track_name,
        )
        fig.savefig(report_dir / f"{gene}_overlay.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        four_gene_panels.append(
            {
                "gene": gene,
                "track_name": track_name,
                "tss": coord.tss,
                "bin_starts": bin_starts,
                "pred_signal": pred_signal,
                "x_attr": window.start + np.arange(len(attr_scores), dtype=np.int64),
                "attr_scores": attr_scores,
                "candidate_intervals": candidate_intervals,
                "predicted_intervals": predicted_intervals,
            }
        )

        cleanup_memory()

    if not metric_rows:
        raise RuntimeError("No genes were successfully processed.")

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["peak_ccre_f1", "peak_ccre_precision", "peak_ccre_recall"],
        ascending=False,
    )
    metrics_df.to_csv(output_dir / "four_gene_metrics.csv", index=False)

    overall_df = pd.DataFrame(
        [
            {
                "genes_evaluated": int(metrics_df["gene"].nunique()),
                "track_index": track_index,
                "track_name": track_name,
                "mean_peak_ccre_precision": float(metrics_df["peak_ccre_precision"].mean()),
                "mean_peak_ccre_recall": float(metrics_df["peak_ccre_recall"].mean()),
                "mean_peak_ccre_f1": float(metrics_df["peak_ccre_f1"].mean()),
                "mean_peak_ccre_jaccard_bp": float(metrics_df["peak_ccre_jaccard_bp"].mean()),
            }
        ]
    )
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    if len(four_gene_panels) == 4:
        ordered_panels = [next(panel for panel in four_gene_panels if panel["gene"] == g) for g in figure_genes]
        fig = plot_four_gene_overview(ordered_panels)
        fig.savefig(report_dir / "four_gene_overview.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print("Saved 4-gene metrics to", output_dir / "four_gene_metrics.csv")
    print("Saved summary to", output_dir / "overall_metrics.csv")
    print("Saved figures to", report_dir)


if __name__ == "__main__":
    main()
