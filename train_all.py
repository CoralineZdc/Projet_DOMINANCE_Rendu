#!/usr/bin/env python
"""Run the batch training launcher from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical batch training launcher."""
    training_dir = Path(__file__).parent / "src" / "training"
    sys.path.insert(0, str(training_dir))

    from train_all_models_all_datasets import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
