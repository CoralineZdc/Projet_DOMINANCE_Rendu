#!/usr/bin/env python
"""Run the model evaluation entry point from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical evaluation entry point."""
    eval_dir = Path(__file__).parent / "src" / "evaluation"
    sys.path.insert(0, str(eval_dir))

    from evaluation import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
