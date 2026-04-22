#!/usr/bin/env python
"""Run the training-log analysis entry point from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical training-log analysis entry point."""
    eval_dir = Path(__file__).parent / "src" / "evaluation"
    sys.path.insert(0, str(eval_dir))

    from training_analysis import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
