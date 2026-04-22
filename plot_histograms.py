#!/usr/bin/env python
"""Run the histogram plotting entry point from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical histogram plotting entry point."""
    eval_dir = Path(__file__).parent / "src" / "evaluation"
    sys.path.insert(0, str(eval_dir))

    from plot_emotion_histograms import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
