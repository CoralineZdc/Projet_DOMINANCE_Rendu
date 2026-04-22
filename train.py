#!/usr/bin/env python
"""Run the single-run training entry point from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical single-model training entry point."""
    training_dir = Path(__file__).parent / "src" / "training"
    sys.path.insert(0, str(training_dir))

    from mainpro_FER import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
