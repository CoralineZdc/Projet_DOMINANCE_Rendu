#!/usr/bin/env python
"""Evaluate every saved run checkpoint found under the runs directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluation.training_analysis import read_log, values_and_epochs


CHECKPOINT_PRIORITY = [
    "best_model.pth",
    "best_model_state.pth",
    "checkpoint.pth",
    "last_model.pth",
]

LOSS_PLOT_NAME = "all_loss_curves.png"
RUN_LOSS_PLOT_NAME = "loss_curves.png"


def find_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-priority checkpoint found in a run directory."""
    for checkpoint_name in CHECKPOINT_PRIORITY:
        candidate = run_dir / checkpoint_name
        if candidate.exists():
            return candidate
    return None


def iter_run_dirs(runs_root: Path):
    """Yield immediate subdirectories that look like saved runs."""
    if not runs_root.exists():
        return
    for child in sorted(runs_root.iterdir()):
        if child.is_dir():
            yield child


def run_evaluation(checkpoint_path: Path, output_dir: Path, extra_args: list[str]) -> None:
    """Invoke the canonical evaluator for one checkpoint."""
    command = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--model",
        str(checkpoint_path),
        "--output_dir",
        str(output_dir),
        *extra_args,
    ]
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def plot_all_loss_curves(run_logs: list[tuple[str, Path]], output_path: Path) -> None:
    """Plot train and public loss curves for every run on shared axes."""
    if not run_logs:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    cmap = plt.get_cmap("tab20")
    plotted = 0

    for index, (run_name, log_path) in enumerate(run_logs):
        try:
            series = read_log(log_path)
        except Exception as exc:
            print(f"Skipping loss plot for {run_name}: {exc}")
            continue

        color = cmap(index % cmap.N)
        xs, ys = values_and_epochs(series["epochs"], series["train_loss"])
        if xs:
            axes[0].plot(xs, ys, color=color, linewidth=1.7, alpha=0.9, label=run_name)
            plotted += 1

        xs, ys = values_and_epochs(series["epochs"], series["public_loss"])
        if xs:
            axes[1].plot(xs, ys, color=color, linewidth=1.7, alpha=0.9)

    if plotted == 0:
        plt.close(fig)
        print("No valid loss curves found to plot.")
        return

    axes[0].set_title("Training Loss Curves")
    axes[0].set_ylabel("train_loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    axes[1].set_title("Public Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("public_loss")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("All Loss Curves Across Saved Runs")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss-curve plot to {output_path}")


def plot_run_loss_curves(run_name: str, log_path: Path, output_path: Path) -> None:
    """Plot train and public loss curves for one run."""
    try:
        series = read_log(log_path)
    except Exception as exc:
        print(f"Skipping loss plot for {run_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    xs, ys = values_and_epochs(series["epochs"], series["train_loss"])
    if xs:
        ax.plot(xs, ys, linewidth=2.0, label="train_loss")

    xs, ys = values_and_epochs(series["epochs"], series["public_loss"])
    if xs:
        ax.plot(xs, ys, linewidth=2.0, label="public_loss")

    if not ax.lines:
        plt.close(fig)
        print(f"Skipping loss plot for {run_name}: no valid loss values found")
        return

    ax.set_title(f"Loss Curves - {run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss-curve plot to {output_path}")


def main() -> None:
    """Parse arguments and evaluate every run checkpoint that exists."""
    parser = argparse.ArgumentParser(description="Evaluate every saved checkpoint under runs/.")
    parser.add_argument("--runs-root", type=str, default="runs", help="Directory that contains saved run folders.")
    parser.add_argument("--output-root", type=str, default="evaluations", help="Directory that will receive evaluation outputs.")
    parser.add_argument("--cuda", action="store_true", help="Pass --cuda to the evaluator when available.")
    parser.add_argument("--cut_size", type=int, default=48, help="Spatial crop size used at evaluation.")
    parser.add_argument("--input_size", type=int, default=0, help="Final model input size after crop (0 uses cut_size).")
    parser.add_argument("--align_faces", action="store_true", help="Enable OpenCV Haar-based face alignment before transforms.")
    parser.add_argument("--public_csv", type=str, default="", help="Optional override CSV for PublicTest split.")
    parser.add_argument("--private_csv", type=str, default="", help="Optional override CSV for PrivateTest split.")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit the number of batches per split for smoke testing.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used for evaluation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint filename override, searched inside each run folder first.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = repo_root / runs_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    extra_args = ["--cut_size", str(args.cut_size), "--batch_size", str(args.batch_size)]
    if args.input_size > 0:
        extra_args.extend(["--input_size", str(args.input_size)])
    if args.align_faces:
        extra_args.append("--align_faces")
    if args.public_csv:
        extra_args.extend(["--public_csv", args.public_csv])
    if args.private_csv:
        extra_args.extend(["--private_csv", args.private_csv])
    if args.max_batches is not None:
        extra_args.extend(["--max_batches", str(args.max_batches)])
    if args.cuda:
        extra_args.append("--cuda")

    evaluated = 0
    skipped = 0
    failed = 0
    run_logs: list[tuple[str, Path]] = []

    for run_dir in iter_run_dirs(runs_root):
        log_path = run_dir / "log.csv"
        if log_path.exists():
            run_logs.append((run_dir.name, log_path))

        checkpoint_path = None
        if args.checkpoint:
            preferred = run_dir / args.checkpoint
            if preferred.exists():
                checkpoint_path = preferred
        if checkpoint_path is None:
            checkpoint_path = find_checkpoint(run_dir)

        if checkpoint_path is None:
            print(f"Skipping {run_dir.name}: no checkpoint found")
            skipped += 1
            continue

        run_output_dir = output_root / run_dir.name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            print(f"Would evaluate {checkpoint_path} -> {run_output_dir}")
            evaluated += 1
            continue

        try:
            run_evaluation(checkpoint_path, run_output_dir, extra_args)
        except subprocess.CalledProcessError as exc:
            print(f"Evaluation failed for {run_dir.name}: {exc}")
            failed += 1
        finally:
            log_path = run_dir / "log.csv"
            if log_path.exists():
                plot_run_loss_curves(run_dir.name, log_path, run_output_dir / RUN_LOSS_PLOT_NAME)

        evaluated += 1

    loss_plot_path = output_root / LOSS_PLOT_NAME
    plot_all_loss_curves(run_logs, loss_plot_path)

    print(f"Completed {evaluated} evaluation(s); failed {failed}; skipped {skipped} run(s) without checkpoints.")


if __name__ == "__main__":
    main()