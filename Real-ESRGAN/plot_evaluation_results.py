import csv
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np


def load_compare_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_lpips = None
            if "avg_lpips" in row and row["avg_lpips"]:
                avg_lpips = float(row["avg_lpips"])
            rows.append(
                {
                    "model": row["model"],
                    "avg_psnr": float(row["avg_psnr"]),
                    "avg_ssim": float(row["avg_ssim"]),
                    "avg_lpips": avg_lpips,
                }
            )
    return rows


def short_model_name(model_name):
    mapping = {
        "Bicubic": "Bicubic",
        "ESRGAN_x4": "ESRGAN",
        "RealESRGAN_x4plus": "x4plus",
        "realesr-general-x4v3": "x4v3",
        "RealESRGAN_x4plus_anime_6B": "anime_6B",
    }
    return mapping.get(model_name, model_name)


def annotate_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_single_metric(metric_key, title, y_label, datasets, model_names, values_table, output_path):
    x = np.arange(len(datasets))
    bar_count = len(values_table[datasets[0]])
    width = 0.8 / max(bar_count, 1)
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), dpi=180)
    colors = ["#2f6fed", "#f08c2e", "#2ca58d", "#7f5af0", "#2cb67d"]

    for model_idx, model_name in enumerate(model_names):
        offset = (model_idx - (bar_count - 1) / 2) * width
        metric_values = [values_table[dataset][model_idx] for dataset in datasets]
        bars = ax.bar(
            x + offset,
            metric_values,
            width=width,
            label=short_model_name(model_name),
            color=colors[model_idx % len(colors)],
        )
        fmt = "{:.4f}" if metric_key == "avg_lpips" else "{:.3f}"
        annotate_bars(ax, bars, fmt)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric_key} plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results from model_compare.csv files.")
    parser.add_argument("--root", type=str, default="evaluation_results", help="Evaluation result root directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "Bsd100"],
        help="Dataset folders under the root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("evaluation_results", "plots"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    rows_by_dataset = {}
    model_names = []
    has_lpips = False

    for dataset in args.datasets:
        csv_path = os.path.join(args.root, dataset, "model_compare.csv")
        rows = load_compare_csv(csv_path)
        rows_by_dataset[dataset] = rows
        if not model_names:
            model_names = [row["model"] for row in rows]
        has_lpips = has_lpips or any(row["avg_lpips"] is not None for row in rows)

    os.makedirs(args.output_dir, exist_ok=True)
    metric_specs = [
        ("avg_psnr", "PSNR Comparison", "PSNR (dB)", "model_compare_psnr.png"),
        ("avg_ssim", "SSIM Comparison", "SSIM", "model_compare_ssim.png"),
    ]
    if has_lpips:
        metric_specs.append(("avg_lpips", "LPIPS Comparison (Lower Better)", "LPIPS", "model_compare_lpips.png"))

    for metric_key, title, y_label, file_name in metric_specs:
        values_table = {}
        for dataset in args.datasets:
            values_table[dataset] = [row[metric_key] for row in rows_by_dataset[dataset]]
        output_path = os.path.join(args.output_dir, file_name)
        plot_single_metric(metric_key, title, y_label, args.datasets, model_names, values_table, output_path)


if __name__ == "__main__":
    main()
