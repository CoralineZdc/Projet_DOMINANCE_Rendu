#!/usr/bin/env python
"""Run a lightweight repository smoke test for local use and CI."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], *, cwd: Path) -> None:
    """Execute a command and fail fast if it exits with an error."""
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def find_checkpoint() -> Path | None:
    """Return the first bundled checkpoint that exists, if any."""
    preferred_runs = [
        REPO_ROOT / "runs" / "FER2013_ResNet18" / "best_model.pth",
        REPO_ROOT / "runs" / "FER2013_ResNet18" / "best_model_state.pth",
        REPO_ROOT / "runs" / "FER2013_ResNet50" / "best_model.pth",
        REPO_ROOT / "runs" / "FER2013_EfficientNet" / "best_model.pth",
        REPO_ROOT / "runs" / "FER2013_MobileFaceNet" / "best_model.pth",
    ]
    for candidate in preferred_runs:
        if candidate.exists():
            return candidate

    for candidate in sorted((REPO_ROOT / "runs").glob("*/best_model.pth")):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    """Run the launcher dry run and, when possible, one evaluation pass."""
    with tempfile.TemporaryDirectory(prefix="vadnet-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "train_all.py"),
                "--datasets",
                "fer2013",
                "--models",
                "resnet18",
                "--output-root",
                str(temp_path / "runs"),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
        )

        checkpoint = find_checkpoint()
        if checkpoint is None:
            print("No bundled checkpoint found; skipping evaluation smoke check.")
            return

        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "evaluate.py"),
                "--model",
                str(checkpoint),
                "--output_dir",
                str(temp_path / "evaluation"),
                "--max_batches",
                "1",
                "--batch_size",
                "1",
            ],
            cwd=REPO_ROOT,
        )


if __name__ == "__main__":
    main()