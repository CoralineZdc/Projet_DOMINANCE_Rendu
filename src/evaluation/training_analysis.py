"""
Unified training log analysis tool.
Consolidated from display_log_losses.py and plot_training_curves.py.
Supports both terminal display and plotting modes.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# ============================================================================
# SHARED UTILITIES
# ============================================================================

MEAN_CANDIDATES = ["pub_rmse_mean", "rmse_mean", "mean"]
VAL_CANDIDATES = ["pub_rmse_val", "rmse_val", "valence", "v"]
ARO_CANDIDATES = ["pub_rmse_aro", "rmse_aro", "arousal", "a"]
DOM_CANDIDATES = ["pub_rmse_dom", "rmse_dom", "dominance", "d"]
EPOCH_CANDIDATES = ["epoch"]


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Find first matching column name from candidates list."""
    names = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def to_float(value: str) -> Optional[float]:
    """Convert string to float, handling errors."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> Optional[int]:
    """Convert string to int via float."""
    f = to_float(value)
    if f is None:
        return None
    return int(f)


def format_float(x: Optional[float], ndigits: int = 6) -> str:
    """Format float for display."""
    return "-" if x is None else f"{x:.{ndigits}f}"


def format_int(x: Optional[int]) -> str:
    """Format int for display."""
    return "-" if x is None else str(x)


def read_log(path: Path) -> Dict:
    """Read training log CSV and extract VAD metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        epoch_col = pick_column(reader.fieldnames, EPOCH_CANDIDATES)
        train_loss_col = pick_column(reader.fieldnames, ["train_loss", "loss"])
        public_loss_col = pick_column(reader.fieldnames, ["public_loss", "val_loss", "valid_loss"])
        mean_col = pick_column(reader.fieldnames, MEAN_CANDIDATES)
        val_col = pick_column(reader.fieldnames, VAL_CANDIDATES)
        aro_col = pick_column(reader.fieldnames, ARO_CANDIDATES)
        dom_col = pick_column(reader.fieldnames, DOM_CANDIDATES)

        if mean_col is None or val_col is None or aro_col is None or dom_col is None:
            raise ValueError(
                f"Missing required columns in {path}. "
                f"Need mean/V/A/D columns (got: {reader.fieldnames})"
            )

        # Store both the full row view and column-wise arrays so plotting and tabular display reuse the same parse.
        rows: List[Dict] = []
        epochs: List[Optional[int]] = []
        train_loss: List[Optional[float]] = []
        public_loss: List[Optional[float]] = []
        mean: List[Optional[float]] = []
        val: List[Optional[float]] = []
        aro: List[Optional[float]] = []
        dom: List[Optional[float]] = []

        for row in reader:
            epochs.append(to_int(row.get(epoch_col, "")) if epoch_col else None)
            train_loss.append(to_float(row.get(train_loss_col, "")) if train_loss_col else None)
            public_loss.append(to_float(row.get(public_loss_col, "")) if public_loss_col else None)
            mean.append(to_float(row.get(mean_col, "")))
            val.append(to_float(row.get(val_col, "")))
            aro.append(to_float(row.get(aro_col, "")))
            dom.append(to_float(row.get(dom_col, "")))

            rows.append({
                "epoch": epochs[-1],
                "mean": mean[-1],
                "v": val[-1],
                "a": aro[-1],
                "d": dom[-1],
            })

    valid_rows = [r for r in rows if r["mean"] is not None]
    if not valid_rows:
        raise ValueError(f"No valid mean values found in {path}")

    best_row = min(valid_rows, key=lambda r: r["mean"])
    last_row = valid_rows[-1]

    return {
        "path": path,
        "rows": rows,
        "best_row": best_row,
        "last_row": last_row,
        "epochs": epochs,
        "train_loss": train_loss,
        "public_loss": public_loss,
        "mean": mean,
        "val": val,
        "aro": aro,
        "dom": dom,
    }


def gather_logs_from_summary(summary_path: Path) -> List[Path]:
    """Extract log paths from grid search summary CSV."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    logs: List[Path] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return logs
        run_dir_col = pick_column(reader.fieldnames, ["run_dir"])
        if run_dir_col is None:
            return logs

        for row in reader:
            # Each summary row points to a run directory that should contain train_log.csv.
            run_dir_raw = str(row.get(run_dir_col, "")).strip()
            if not run_dir_raw:
                continue
            run_dir = Path(run_dir_raw.replace("\\", "/"))
            log_path = run_dir / "train_log.csv"
            if log_path.exists():
                logs.append(log_path)

    return logs


def gather_run_entries_from_summary(summary_path: Path) -> List[Tuple[str, Path]]:
    """Extract run labels and log paths from summary CSV."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    entries: List[Tuple[str, Path]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return entries

        run_dir_col = pick_column(reader.fieldnames, ["run_dir"])
        run_id_col = pick_column(reader.fieldnames, ["run_id"])
        dataset_col = pick_column(reader.fieldnames, ["dataset"])
        status_col = pick_column(reader.fieldnames, ["status"])

        if run_dir_col is None:
            raise ValueError(f"Missing run_dir column in {summary_path}")

        for row in reader:
            if status_col and str(row.get(status_col, "")).strip().lower() not in {"", "ok"}:
                continue

            run_dir_raw = str(row.get(run_dir_col, "")).strip()
            if not run_dir_raw:
                continue

            run_dir = Path(run_dir_raw.replace("\\", "/"))
            log_path = run_dir / "train_log.csv"
            if not log_path.exists():
                continue

            run_id = str(row.get(run_id_col, "")).strip() if run_id_col else ""
            dataset = str(row.get(dataset_col, "")).strip() if dataset_col else ""
            # Prefer a human-readable label when the summary provides run_id/dataset columns.
            label = run_dir.name
            if run_id and dataset:
                label = f"{run_id}:{dataset}"
            elif dataset:
                label = dataset
            elif run_id:
                label = f"run_{run_id}"

            entries.append((label, log_path))

    return entries


# ============================================================================
# TERMINAL DISPLAY MODE
# ============================================================================

def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print nicely formatted table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def print_run_summary(log_infos: Sequence[Dict]) -> None:
    """Display summary of best and last epoch metrics for each run."""
    rows: List[List[str]] = []
    for info in log_infos:
        path = info["path"]
        best = info["best_row"]
        last = info["last_row"]
        rows.append([
            Path(path).parent.name,
            format_int(best["epoch"]),
            format_float(best["mean"]),
            format_float(best["v"]),
            format_float(best["a"]),
            format_float(best["d"]),
            format_int(last["epoch"]),
            format_float(last["mean"]),
            format_float(last["v"]),
            format_float(last["a"]),
            format_float(last["d"]),
        ])

    # Keep the summary compact so multiple runs can be compared at a glance.
    headers = [
        "run",
        "best_ep",
        "best_mean",
        "best_V",
        "best_A",
        "best_D",
        "last_ep",
        "last_mean",
        "last_V",
        "last_A",
        "last_D",
    ]
    print_table(headers, rows)


def print_per_epoch(log_info: Dict, tail: int) -> None:
    """Display per-epoch metrics."""
    rows = [r for r in log_info["rows"] if r["mean"] is not None]
    if tail > 0:
        rows = rows[-tail:]

    table_rows: List[List[str]] = []
    for r in rows:
        table_rows.append([
            format_int(r["epoch"]),
            format_float(r["mean"]),
            format_float(r["v"]),
            format_float(r["a"]),
            format_float(r["d"]),
        ])

    print(f"\n[{log_info['path']}]")
    print_table(["epoch", "mean", "V", "A", "D"], table_rows)


# ============================================================================
# PLOTTING MODE
# ============================================================================

def values_and_epochs(epochs: List[Optional[int]], values: List[Optional[float]]):
    """Extract valid epoch-value pairs."""
    xs = []
    ys = []
    for epoch, value in zip(epochs, values):
        if epoch is None or value is None:
            continue
        xs.append(epoch)
        ys.append(value)
    return xs, ys


def plot_compare_mean(entries: List[Tuple[str, Path]], out_path: Path, title: str, show: bool) -> None:
    """Plot mean RMSE comparison across multiple runs."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    plotted = 0
    for label, log_path in entries:
        # Plot only the shared mean metric so run-to-run comparisons stay readable.
        series = read_log(log_path)
        xs, ys = values_and_epochs(series["epochs"], series["mean"])
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=2.0, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("No valid mean curves found to compare.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_single_log(log_path: Path, out_path: Path, title: str, show: bool) -> None:
    """Plot training curves for a single run."""
    series = read_log(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: optimization losses.
    xs, ys = values_and_epochs(series["epochs"], series["train_loss"])
    if xs:
        axes[0].plot(xs, ys, label="train_loss", linewidth=2.0)

    xs, ys = values_and_epochs(series["epochs"], series["public_loss"])
    if xs:
        axes[0].plot(xs, ys, label="public_loss", linewidth=2.0)

    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # Bottom panel: evaluation metrics on the VAD targets.
    xs, ys = values_and_epochs(series["epochs"], series["mean"])
    if xs:
        axes[1].plot(xs, ys, label="mean", linewidth=2.2)

    xs, ys = values_and_epochs(series["epochs"], series["val"])
    if xs:
        axes[1].plot(xs, ys, label="V", linewidth=1.8)

    xs, ys = values_and_epochs(series["epochs"], series["aro"])
    if xs:
        axes[1].plot(xs, ys, label="A", linewidth=1.8)

    xs, ys = values_and_epochs(series["epochs"], series["dom"])
    if xs:
        axes[1].plot(xs, ys, label="D", linewidth=1.8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved curve plot to {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified training log analysis: display metrics or plot curves."
    )
    parser.add_argument(
        "--mode",
        choices=["display", "plot"],
        default="display",
        help="Display metrics in terminal or plot to PNG (default: display).",
    )

    # Shared arguments
    parser.add_argument(
        "logs",
        nargs="*",
        help="Paths to log CSV files (display mode) or single log (plot mode).",
    )

    # Display mode arguments
    parser.add_argument(
        "--from-summary",
        type=str,
        default="",
        help="Load logs from grid search summary CSV.",
    )
    parser.add_argument(
        "--per-epoch",
        action="store_true",
        help="Display mode: also print per-epoch metrics.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=15,
        help="Display mode: show last N epochs (0 = all).",
    )

    # Plot mode arguments
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Plot mode: output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Plot mode: display plot window.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot mode: custom title.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="training_curve",
        help="Plot mode with --from-summary: output filename prefix.",
    )
    parser.add_argument(
        "--compare-mean",
        action="store_true",
        help="Plot mode: overlay mean RMSE from all runs.",
    )
    parser.add_argument(
        "--compare-out",
        type=str,
        default="",
        help="Plot mode: output path for comparison plot.",
    )

    args = parser.parse_args()

    # ===== DISPLAY MODE =====
    if args.mode == "display":
        # Display mode prints tables; plot mode writes PNG files.
        log_paths: List[Path] = [Path(p) for p in args.logs]

        if args.from_summary:
            log_paths.extend(gather_logs_from_summary(Path(args.from_summary)))

        # Remove duplicates preserving order
        unique_paths: List[Path] = []
        seen = set()
        for p in log_paths:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                unique_paths.append(p)
                seen.add(key)

        if not unique_paths:
            raise SystemExit("No logs provided. Pass log paths or --from-summary.")

        infos: List[Dict] = []
        for p in unique_paths:
            infos.append(read_log(p))

        print_run_summary(infos)

        if args.per_epoch:
            for info in infos:
                print_per_epoch(info, args.tail)

    # ===== PLOT MODE =====
    elif args.mode == "plot":
        if args.from_summary:
            summary_path = Path(args.from_summary)

            if args.compare_mean:
                # Generate one overlay figure before writing the per-run figures.
                entries = gather_run_entries_from_summary(summary_path)
                if not entries:
                    raise SystemExit("No valid run entries found to compare.")
                default_out = summary_path.parent / "compare_mean_rmse.png"
                out_path = Path(args.compare_out) if args.compare_out else default_out
                title = args.title.strip() or f"Mean RMSE Comparison ({summary_path.parent.name})"
                plot_compare_mean(entries, out_path, title, args.show)

            logs = gather_logs_from_summary(summary_path)
            if not logs:
                raise SystemExit("No train_log.csv files found from summary.")

            for log_path in logs:
                run_name = log_path.parent.name
                out_path = log_path.parent / f"{args.prefix}.png"
                title = args.title.strip() or run_name
                plot_single_log(log_path, out_path, title, show=False)
            print(f"Done. Plotted {len(logs)} runs.")
        else:
            if not args.logs:
                raise SystemExit("Provide a log path, or use --from-summary.")

            log_path = Path(args.logs[0])
            out_path = Path(args.out) if args.out else log_path.with_suffix(".png")
            title = args.title.strip() or log_path.parent.name
            plot_single_log(log_path, out_path, title, args.show)


if __name__ == "__main__":
    main()
