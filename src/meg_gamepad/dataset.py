"""Site-specific dataset description.

The experimental design (which conditions exist, what each one means) is fixed
and lives in :mod:`config`. Everything that changes from site to site lives
here, in a JSON manifest stored next to the recordings:

* the file name of each run, which no two sites will spell the same way;
* the STI101 bit values the buttons raise, which depend on local wiring;
* the mains frequency, if it cannot be read from the file header.

A site therefore adds a directory and a manifest. It changes no code.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONDITIONS, CONDITION_BY_KEY

log = logging.getLogger(__name__)

MANIFEST_NAME = "dataset.json"

#: Keywords used to guess which file is which run, if a site has no manifest.
#: Each entry is (condition key, patterns that must all appear in the name).
DISCOVERY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", ("baseline",)),
    ("baseline", ("off",)),
    ("passive", ("passive",)),
    ("passive", ("not", "moving")),
    ("playing", ("playing",)),
    ("playing", ("button",)),
    ("motion", ("motion",)),
    ("motion", ("moving",)),
)


class DatasetError(RuntimeError):
    """The dataset directory does not describe a usable set of runs."""


@dataclass
class Dataset:
    """One site's recordings, resolved to absolute paths."""

    site: str
    root: Path
    #: condition key -> absolute path of the recording
    files: dict[str, Path]
    #: condition key -> STI101 bit values, overriding the design defaults
    button_codes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: mains frequency in Hz; None means "take it from the file header"
    line_freq: float | None = None
    description: str = ""
    discovered: bool = False

    def codes_for(self, key: str) -> tuple[int, ...]:
        """Button bit values for a condition, from the manifest or the design."""
        if key in self.button_codes:
            return self.button_codes[key]
        return CONDITION_BY_KEY[key].button_codes

    def missing(self) -> list[str]:
        return [c.key for c in CONDITIONS if c.key not in self.files]


def _resolve(root: Path, name: str, key: str) -> Path:
    path = (root / name).expanduser()
    if not path.is_absolute():
        path = root / name
    if not path.exists():
        raise DatasetError(
            f"{MANIFEST_NAME} lists '{name}' for the '{key}' run, "
            f"but {path} does not exist."
        )
    return path


def load_manifest(root: Path) -> Dataset:
    """Read ``dataset.json`` from a dataset directory."""
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{manifest_path} is not valid JSON: {exc}") from exc

    runs = raw.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise DatasetError(f"{manifest_path} has no 'runs' object.")

    known = {c.key for c in CONDITIONS}
    unknown = set(runs) - known
    if unknown:
        raise DatasetError(
            f"{manifest_path} names unknown runs {sorted(unknown)}. "
            f"The known runs are {sorted(known)}."
        )

    files = {key: _resolve(root, name, key) for key, name in runs.items()}
    codes = {
        key: tuple(int(v) for v in vals)
        for key, vals in (raw.get("button_codes") or {}).items()
        if key in known
    }
    line_freq = raw.get("line_freq")
    return Dataset(
        site=str(raw.get("site") or root.name),
        root=root,
        files=files,
        button_codes=codes,
        line_freq=float(line_freq) if line_freq else None,
        description=str(raw.get("description") or ""),
    )


def discover(root: Path) -> Dataset:
    """Guess the run of each ``.fif`` file from its name.

    This is a convenience for a site that has not yet written a manifest. It
    is deliberately conservative: it refuses an ambiguous match rather than
    guessing, because a mislabelled run would invert the conclusions.
    """
    candidates = sorted(p for p in root.glob("*.fif") if p.is_file())
    if not candidates:
        raise DatasetError(f"No .fif files in {root}.")

    matches: dict[str, list[Path]] = {}
    for path in candidates:
        stem = re.sub(r"[^a-z0-9]+", " ", path.stem.lower())
        hits = {
            key
            for key, patterns in DISCOVERY_RULES
            if all(p in stem for p in patterns)
        }
        if len(hits) == 1:
            matches.setdefault(hits.pop(), []).append(path)
        elif len(hits) > 1:
            raise DatasetError(
                f"The name '{path.name}' matches several runs {sorted(hits)}. "
                f"Write a {MANIFEST_NAME} to state which run it is."
            )

    duplicated = {k: v for k, v in matches.items() if len(v) > 1}
    if duplicated:
        detail = "; ".join(
            f"{k}: {', '.join(p.name for p in v)}" for k, v in duplicated.items()
        )
        raise DatasetError(
            f"Several files match the same run ({detail}). "
            f"Write a {MANIFEST_NAME} to state which file is which."
        )

    files = {k: v[0] for k, v in matches.items()}
    log.warning(
        "No %s in %s. Guessed the runs from the file names: %s",
        MANIFEST_NAME,
        root,
        ", ".join(f"{k}={v.name}" for k, v in sorted(files.items())),
    )
    return Dataset(site=root.name, root=root, files=files, discovered=True)


def load_dataset(root: Path) -> Dataset:
    """Load a dataset directory, by manifest if there is one, else by name."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(
            f"{root} is not a directory. Pass --data-dir, or put the recordings "
            f"in a directory with a {MANIFEST_NAME}."
        )
    dataset = (
        load_manifest(root) if (root / MANIFEST_NAME).exists() else discover(root)
    )
    missing = dataset.missing()
    if missing:
        raise DatasetError(
            f"{root} is missing the {missing} run(s). "
            f"The analysis compares all of {[c.key for c in CONDITIONS]}."
        )
    return dataset


def write_template(root: Path, site: str | None = None) -> Path:
    """Write a manifest listing the ``.fif`` files found, for a site to edit."""
    root = Path(root).expanduser().resolve()
    found = sorted(p.name for p in root.glob("*.fif"))
    runs = {c.key: "" for c in CONDITIONS}
    try:
        for key, path in discover(root).files.items():
            runs[key] = path.name
    except DatasetError:
        pass
    manifest = {
        "site": site or root.name,
        "description": "",
        "runs": runs,
        "button_codes": {
            c.key: list(c.button_codes) for c in CONDITIONS if c.button_codes
        },
        "_files_present": found,
        "_help": (
            "runs maps each condition to a file name in this directory. "
            "button_codes gives the STI101 bit values each button raises; "
            "omit a key to use the default. line_freq overrides the mains "
            "frequency in the file header."
        ),
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
