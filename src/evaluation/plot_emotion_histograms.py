import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMS = ["Valence", "Arousal", "Dominance"]
DEFAULT_BINS = np.arange(-2.0, 2.0 + 0.5, 0.5)


def find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    mapping = {str(name).strip().lower(): str(name) for name in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in mapping:
            return mapping[key]
    return None


def load_dimension_columns(csv_path: Path) -> dict[str, str]:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")

    column_map: dict[str, str] = {}
    aliases = {
        "Valence": ["Valence", "valence", "V", "v"],
        "Arousal": ["Arousal", "arousal", "A", "a"],
        "Dominance": ["Dominance", "dominance", "D", "d"],
    }

    for dim, candidates in aliases.items():
        column = find_column(frame.columns, candidates)
        if column is None:
            raise ValueError(f"Missing {dim} column in {csv_path}")
        column_map[dim] = column

    return column_map


def plot_histograms(csv_path: Path, output_dir: Path, bins: np.ndarray) -> Path:
    frame = pd.read_csv(csv_path)
    column_map = load_dimension_columns(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, dim in zip(axes, DIMS):
        values = pd.to_numeric(frame[column_map[dim]], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No numeric values found for {dim} in {csv_path}")

        ax.hist(values, bins=bins, edgecolor="black", linewidth=0.8, color="#4C78A8", alpha=0.9)
        ax.set_title(dim)
        ax.set_xlabel("Score")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(bins)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Count")
    fig.suptitle(csv_path.stem)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}_histograms.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def discover_csvs(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*.csv")
        if path.is_file() and path.name.lower().endswith(".csv")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw Valence, Arousal, and Dominance histograms for VAD CSV datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory containing dataset CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/emotion_histograms",
        help="Directory where the histogram images will be saved.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="",
        help="Optional comma-separated list of CSV files to plot instead of scanning the input directory.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.5,
        help="Histogram bin width.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=-2.0,
        help="Lower bound of the histogram range.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=2.0,
        help="Upper bound of the histogram range.",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[int, list[str]]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if str(args.files).strip():
        csv_paths = [Path(item.strip()) for item in str(args.files).split(",") if item.strip()]
    else:
        csv_paths = discover_csvs(input_dir)

    if not csv_paths:
        raise SystemExit(f"No CSV files found in {input_dir}")

    bins = np.arange(args.min_value, args.max_value + args.bin_width, args.bin_width)
    if bins[-1] < args.max_value:
        bins = np.append(bins, args.max_value)

    saved = 0
    skipped: list[str] = []

    for csv_path in csv_paths:
        try:
            out_path = plot_histograms(csv_path, output_dir, bins)
            print(f"Saved {out_path}")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{csv_path}: {exc}")

    return saved, skipped

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    saved, skipped = run(args)

    print(f"Plotted {saved} dataset(s).")
    if skipped:
        print("Skipped datasets:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
