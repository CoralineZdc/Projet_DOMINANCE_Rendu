#!/usr/bin/env python
"""Run the consolidated dataset preparation pipeline from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical dataset preparation entry point."""
    data_dir = Path(__file__).parent / "src" / "data"
    sys.path.insert(0, str(data_dir))

    from prepare_all_datasets import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
