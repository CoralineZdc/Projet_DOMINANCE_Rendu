#!/usr/bin/env python
"""Run the face preprocessing entry point from the repository root."""

import sys
from pathlib import Path


def main() -> None:
    """Import and run the canonical face preprocessing entry point."""
    data_dir = Path(__file__).parent / "src" / "data"
    sys.path.insert(0, str(data_dir))

    from preprocess_faces import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
