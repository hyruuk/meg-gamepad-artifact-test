#!/usr/bin/env python3
"""Run the full analysis and write the HTML report.

Usable without installing the package:  python scripts/run_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meg_gamepad.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
