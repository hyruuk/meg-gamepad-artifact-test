"""Reading the raw MEGIN files and bringing all runs onto a common footing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from .config import CONDITIONS, PARAMS, Condition
from .dataset import Dataset, load_dataset

log = logging.getLogger(__name__)


@dataclass
class Run:
    """One cropped recording plus the bookkeeping the report needs."""

    condition: Condition
    path: "Path"
    raw: mne.io.BaseRaw
    #: header facts captured before cropping
    orig_duration: float
    meas_date: str
    sfreq: float
    highpass: float
    lowpass: float
    line_freq: float | None
    n_projs: int
    header_bads: list[str]
    events: np.ndarray

    @property
    def key(self) -> str:
        return self.condition.key

    @property
    def label(self) -> str:
        return self.condition.label


def channel_site(name: str) -> str:
    """MEGIN sensor-site key: ``MEG0112`` and ``MEG0113`` share site ``011``.

    Used to collapse each planar-gradiometer pair onto the single helmet
    position it shares with its magnetometer.
    """
    return name[3:6]


def load_run(condition: Condition, dataset: Dataset, params=PARAMS) -> Run:
    path = dataset.files[condition.key]
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    info = raw.info
    orig_duration = raw.n_times / info["sfreq"]
    meas_date = (
        info["meas_date"].strftime("%Y-%m-%d %H:%M:%S UTC")
        if info["meas_date"] is not None
        else "unknown"
    )

    events = _find_button_events(raw, dataset.codes_for(condition.key))

    tmax = params.crop_tmin + params.crop_duration
    if orig_duration < tmax:
        raise ValueError(
            f"{condition.key}: only {orig_duration:.1f} s long, "
            f"need {tmax:.1f} s for the common window"
        )
    raw.crop(tmin=params.crop_tmin, tmax=tmax)

    # Event samples from find_events are absolute (they include first_samp), and
    # cropping shifts first_samp by the same amount -- so the indices stay valid
    # and must NOT be re-based. Events are kept in full here; each analysis uses a
    # different epoch window, so mne.Epochs is left to drop the ones that do not
    # fit its own window rather than applying the widest margin to everything.
    if len(events):
        lo = raw.first_samp
        hi = raw.first_samp + raw.n_times
        events = events[(events[:, 0] >= lo) & (events[:, 0] < hi)]

    return Run(
        condition=condition,
        path=path,
        raw=raw,
        orig_duration=orig_duration,
        meas_date=meas_date,
        sfreq=float(info["sfreq"]),
        highpass=float(info["highpass"]),
        lowpass=float(info["lowpass"]),
        line_freq=info["line_freq"],
        n_projs=len(info["projs"]),
        header_bads=list(info["bads"]),
        events=events,
    )


def _find_button_events(
    raw: mne.io.BaseRaw, button_codes: tuple[int, ...]
) -> np.ndarray:
    """Button onsets from STI101, restricted to the given button bits.

    The controller's buttons reach the MEG as TTL bits on STI101. Two buttons
    were used, so composite values (both bits high) appear as well; we decode
    bitwise rather than trusting the composite integer.
    """
    if "STI101" not in raw.ch_names or not button_codes:
        return np.empty((0, 3), dtype=int)
    try:
        events = mne.find_events(
            raw, stim_channel="STI101", shortest_event=1, verbose="ERROR"
        )
    except ValueError:
        return np.empty((0, 3), dtype=int)
    if not len(events):
        return events

    # Keep an onset if it raises any of this condition's button bits.
    keep = []
    for samp, prev, cur in events:
        newly_set = int(cur) & ~int(prev)
        for code in button_codes:
            if newly_set & code:
                keep.append((samp, 0, code))
                break
    if not keep:
        return np.empty((0, 3), dtype=int)
    out = np.array(sorted(keep), dtype=int)
    # Collapse onsets closer together than 50 ms (switch bounce / near-simultaneous
    # presses) so the average is not contaminated by double-counting.
    sep = np.diff(out[:, 0], prepend=out[0, 0] - 10_000)
    return out[sep > 0.05 * raw.info["sfreq"]]


def load_all(dataset: Dataset, params=PARAMS) -> dict[str, Run]:
    runs: dict[str, Run] = {}
    for cond in CONDITIONS:
        log.info("loading %s from %s", cond.key, dataset.files[cond.key].name)
        runs[cond.key] = load_run(cond, dataset, params=params)
    _check_comparable(runs)
    return runs


def resolve_line_freq(runs: dict[str, Run], dataset: Dataset) -> float:
    """Mains frequency for this site.

    The manifest wins if it states one. Otherwise the value comes from the file
    headers, which MEGIN systems fill in. The last resort is 60 Hz, and the
    analysis says so in the log when it has to guess.
    """
    if dataset.line_freq:
        return float(dataset.line_freq)
    from_header = {r.line_freq for r in runs.values() if r.line_freq}
    if len(from_header) == 1:
        return float(next(iter(from_header)))
    if len(from_header) > 1:
        log.warning(
            "The files disagree about the mains frequency %s. Using %s Hz.",
            sorted(from_header),
            min(from_header),
        )
        return float(min(from_header))
    log.warning("No mains frequency in the files or the manifest. Using 60 Hz.")
    return 60.0


def _check_comparable(runs: dict[str, Run]) -> list[str]:
    """Warn loudly if the runs are not acquired identically.

    Every cross-run comparison in this report assumes matched acquisition; if
    that fails the dB numbers are meaningless, so it is checked rather than
    assumed.
    """
    warnings: list[str] = []
    ref = next(iter(runs.values()))
    for key, run in runs.items():
        for attr in ("sfreq", "highpass", "lowpass"):
            if not np.isclose(getattr(run, attr), getattr(ref, attr)):
                warnings.append(
                    f"{key}: {attr}={getattr(run, attr)} differs from "
                    f"{ref.key}'s {getattr(ref, attr)}"
                )
        if run.raw.ch_names != ref.raw.ch_names:
            warnings.append(f"{key}: channel set differs from {ref.key}")
    for w in warnings:
        log.warning("acquisition mismatch -- %s", w)
    return warnings
