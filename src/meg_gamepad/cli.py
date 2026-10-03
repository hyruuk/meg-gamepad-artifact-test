"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .analysis import compute_all
from .config import CACHE_PATH, DATA_ROOT, DEFAULT_SITE, REPORT_DIR
from .dataset import DatasetError, load_dataset, write_template
from .report import write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="meg-gamepad",
        description=(
            "Analyse a set of MEG gamepad artifact test runs and write an HTML "
            "report. A dataset is a directory of recordings with a dataset.json "
            "that says which file is which run."
        ),
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--site",
        default=None,
        help=f"dataset directory under {DATA_ROOT} (default: {DEFAULT_SITE})",
    )
    src.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="path to a dataset directory, for data kept outside this repo",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML path (default: reports/<site>.html)",
    )
    ap.add_argument(
        "--init-manifest",
        action="store_true",
        help="write a dataset.json template for the dataset directory, then exit",
    )
    ap.add_argument(
        "--cache",
        action="store_true",
        help=(
            "reuse (and write) a metrics cache. Off by default: a full recompute "
            "takes about the same time as reloading the cache, so it only pays "
            "off on larger datasets or when iterating on the report text alone"
        ),
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("mne").setLevel(logging.ERROR)

    root = args.data_dir or (DATA_ROOT / (args.site or DEFAULT_SITE))

    if args.init_manifest:
        path = write_template(root)
        print(f"Wrote {path}. Fill in the 'runs' names and run the analysis again.")
        return 0

    try:
        dataset = load_dataset(root)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nTo start a new site: put the recordings in a directory, run\n"
            "  meg-gamepad --data-dir <that directory> --init-manifest\n"
            "then edit the dataset.json it writes.",
            file=sys.stderr,
        )
        return 2

    out = args.out or (REPORT_DIR / f"{dataset.site}.html")
    cache = CACHE_PATH if args.cache else None

    results = compute_all(dataset, cache=cache)
    path = write_report(results, out)
    print(f"\nReport written to {path}")
    for key, v in results.verdicts.items():
        print(f"  {key:9s} -> {v['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
