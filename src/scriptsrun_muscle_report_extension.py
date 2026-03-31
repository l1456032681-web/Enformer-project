from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.attribution import (
    attribution_peaks_to_intervals,
    enformer_bins_to_genome_coords,
    gradient_x_input_for_track,
    reduce_attribution,
    smooth_scores,
)
from src.encode import compute_f1, compute_precision_recall, filter_intervals, interval_jaccard, load_ccre_bed
from src.enformer import load_enformer, predict_tracks, prepare_sequence_tensor
from src.genes import fetch_gene_coordinates_mygene
from src.genome import ensure_uncompressed_fasta, get_fasta_reader, load_gene_symbols, make_gene_window, fetch_sequence, pad_sequence, validate_sequence_length
from src.plotting import plot_four_gene_overview, plot_gene_overlay
from src.targets import load_targets


PROJECT_DIR = Path(".")
OUTPUT_DIR = PROJECT_DIR / "outputs_muscle_report"
FIG_DIR = OUTPUT_DIR / "figures"
CCRE_BED = Path("/content/drive/MyDrive/enformer_data/ccre.bed")
HG38_GZ = Path("/content/drive/MyDrive/enformer_data/hg38.fa.gz")
GENE_CSV = PROJECT_DIR / "enformer_project_gene_list.csv"
TARGETS_TXT = PROJECT_DIR / "targets_human.txt"

WINDOW_SIZE = 196_608
TRACK_INDEX = 116   # DNASE:skeletal muscle myoblast
FOUR_GENES = ["ACTA1", "MYH7", "TNNT1", "DES"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    genes = load_gene_symbols(GENE_CSV)
    coords = fetch_gene_coordinates_mygene(genes)
    coord_map = {c.gene: c for c in coords}

    fasta_path = ensure_uncompressed_fasta(HG38_GZ, PROJECT_DIR / "hg38.fa")
    fasta = get_fasta_reader(fasta_path)
    ccres = load_ccre_bed(CCRE_BED)

    targets = load_targets(TARGETS_TXT)
    track_name = str(targets.loc[targets["index"].astype(int) == TRACK_INDEX, "description"].iloc[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_enformer(device=device)

    metric_rows = []
    four_gene_panels = []

    for gene in genes:
        coord = coord_map.get(gene)
        if coord is None:
            continue

        window = make_gene_window(gene, coord.chrom, coord.tss, window_size=WINDOW_SIZE)
        seq = fetch_sequence(fasta, window.chrom, window.start, window.end)
        seq = pad_sequence(seq, WINDOW_SIZE)
        validate_sequence_length(seq, WINDOW_SIZE)

        tokens = prepare_sequence_tensor(seq, device=device)
        preds = predict_tracks(model, tokens)
        preds_human = preds["human"] if isinstance(preds, dict) else preds

        track_values = preds_human[0, :, TRACK_INDEX].detach().cpu().numpy()
        bin_starts = enformer_bins_to_genome_coords(len(track_values), window.start)

        center_bin = int(np.argmax(track_values))
        attr = gradient_x_input_for_track(
            model=model,
            tokens=tokens,
            track_index=TRACK_INDEX,
            bin_index=center_bin,
            head="human",
        )

        attr_scores = reduce_attribution(attr[0], method="sum_abs").detach().cpu().numpy()
        attr_scores = smooth_scores(attr_scores, window=128)

        predicted_intervals = attribution_peaks_to_intervals(
            chrom=window.chrom,
            window_start=window.start,
            scores=attr_scores,
            threshold_quantile=0.995,
            min_width=64,
        )

        candidate_intervals = filter_intervals(ccres, window.chrom, window.start, window.end)

        precision, recall = compute_precision_recall(predicted_intervals, candidate_intervals)
        f1 = compute_f1(precision, recall)
        jacc = interval_jaccard(predicted_intervals, candidate_intervals)

        metric_rows.append(
            {
                "gene": gene,
                "chrom": window.chrom,
                "window_start": window.start,
                "window_end": window.end,
                "track_index": TRACK_INDEX,
                "track_name": track_name,
                "n_predicted_peaks": len(predicted_intervals),
                "n_candidate_ccres": len(candidate_intervals),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard_bp": jacc,
                "max_pred_bin": int(np.argmax(track_values)),
                "max_pred_coord": int(bin_starts[int(np.argmax(track_values))]),
            }
        )

        if gene in FOUR_GENES:
            fig = plot_gene_overlay(
                gene=gene,
                chrom=window.chrom,
                window_start=window.start,
                bin_starts=bin_starts,
                pred_signal=track_values,
                attribution_scores=attr_scores,
                candidate_intervals=candidate_intervals,
                predicted_intervals=predicted_intervals,
                track_name=track_name,
            )
            fig.savefig(FIG_DIR / f"{gene}_overlay.png", dpi=220, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)

            four_gene_panels.append(
                {
                    "gene": gene,
                    "track_name": track_name,
                    "bin_starts": bin_starts,
                    "pred_signal": track_values,
                    "x_attr": window.start + np.arange(len(attr_scores), dtype=np.int64),
                    "attr_scores": attr_scores,
                    "candidate_intervals": candidate_intervals,
                    "predicted_intervals": predicted_intervals,
                }
            )

        del tokens, preds, preds_human, attr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metric_rows).sort_values(["f1", "precision", "recall"], ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "all_180_gene_metrics.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "genes_evaluated": int(metrics_df["gene"].nunique()),
                "mean_precision": float(metrics_df["precision"].mean()),
                "mean_recall": float(metrics_df["recall"].mean()),
                "mean_f1": float(metrics_df["f1"].mean()),
                "mean_jaccard_bp": float(metrics_df["jaccard_bp"].mean()),
                "track_index": TRACK_INDEX,
                "track_name": track_name,
            }
        ]
    )
    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)

    if len(four_gene_panels) == 4:
        fig = plot_four_gene_overview(four_gene_panels)
        fig.savefig(FIG_DIR / "four_gene_overview.png", dpi=220, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    print("Done.")
    print("Metrics:", OUTPUT_DIR / "all_180_gene_metrics.csv")
    print("Figures:", FIG_DIR)


if __name__ == "__main__":
    main()