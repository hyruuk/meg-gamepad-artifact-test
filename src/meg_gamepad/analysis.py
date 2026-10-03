"""Driver: compute every measure once, cache it, and grade the result."""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

from . import events as ev
from . import metrics as mx
from .config import (
    BASELINE_KEY,
    CONDITION_BY_KEY,
    DATA_ROOT,
    DEFAULT_SITE,
    bands_for,
    CACHE_PATH,
    CONDITIONS,
    NON_BASELINE,
    PARAMS,
    SENSORS,
    SEVERITY_ORDER,
    VERDICT_RULES,
    Params,
)
from .dataset import Dataset, load_dataset
from .loading import Run, load_all, resolve_line_freq

log = logging.getLogger(__name__)


@dataclass
class Results:
    """Everything the report needs, free of open file handles."""

    params: Params
    site: str
    dataset_root: str
    line_freq: float
    bands_def: tuple
    inventory: list[dict]
    acquisition_warnings: list[str]
    psds: dict
    bands: dict
    amplitudes: dict
    sliding: dict
    lines: dict
    source_distance: dict
    rejection: dict
    quality: dict
    button: dict
    fine: dict
    locked: dict
    tfr: dict
    motion: dict
    verdicts: dict
    #: sensor positions for topography, keyed by sensor type
    topo_info: dict
    mag_names: list[str]
    mne_version: str
    computed_at: str
    elapsed_s: float


def _isi_median(run: Run) -> float | None:
    """Median interval between button presses, in seconds.

    The press rate controls what the event-locked analyses can resolve: presses
    that follow each other closely leave no pre-stimulus window that is free of
    the previous press.
    """
    if len(run.events) < 2:
        return None
    return float(np.median(np.diff(run.events[:, 0])) / run.sfreq)


def _inventory(runs: dict[str, Run]) -> list[dict]:
    rows = []
    for cond in CONDITIONS:
        run = runs[cond.key]
        rows.append(
            {
                "key": cond.key,
                "label": cond.label,
                "description": cond.description,
                "filename": run.path.name,
                "gamepad_on": cond.gamepad_on,
                "meas_date": run.meas_date,
                "orig_duration": run.orig_duration,
                "analysed_duration": run.raw.n_times / run.sfreq,
                "sfreq": run.sfreq,
                "highpass": run.highpass,
                "lowpass": run.lowpass,
                "line_freq": run.line_freq,
                "n_projs": run.n_projs,
                "header_bads": run.header_bads,
                "n_events": int(len(run.events)),
                "isi_median": _isi_median(run),
                "color": cond.color,
            }
        )
    return rows


def _severity(value: float, minor: float, material: float) -> str:
    if value < minor:
        return "negligible"
    if value < material:
        return "minor"
    return "material"


def _amp_change_db(amplitudes: dict, window: str, key: str, ref: str) -> tuple[float, str]:
    """Worst-across-sensor-type median RMS change vs `ref`, in power dB."""
    worst, worst_sensor = -np.inf, ""
    for sensor in SENSORS:
        vals = amplitudes[sensor.key][window]
        ratio = np.median(vals[key]) / max(np.median(vals[ref]), mx.EPS)
        val = float(mx.db_amp(ratio))
        if val > worst:
            worst, worst_sensor = val, sensor.label
    return worst, worst_sensor


def compute_verdicts(res: dict) -> dict:
    """Grade each condition against the published rule table in config.

    Each condition is graded against its own reference run (see
    ``Condition.reference``): the static conditions against the no-controller
    baseline, the moving condition against the passive run, so that "movement"
    is not confounded with twelve minutes of environmental drift.
    """
    verdicts: dict[str, dict] = {}
    for cond in NON_BASELINE:
        key, ref = cond.key, cond.reference
        measures: dict[str, dict] = {}

        for mkey, window in (
            ("analysis_band", "analysis"),
            ("high_band", "high"),
            ("drift", "drift"),
        ):
            value, sensor_label = _amp_change_db(res["amplitudes"], window, key, ref)
            measures[mkey] = {"value": value, "detail": f"worst on {sensor_label.lower()}"}

        # Narrowband lines: prominence gain over this condition's own reference,
        # restricted to non-powerline peaks above the physiological band.
        best, detail = 0.0, "none above threshold"
        for sensor in SENSORS:
            for line in res["lines"][sensor.key]["lines"]:
                if line["klass"] != "instrument candidate":
                    continue
                gain = line["prominence"][key] - line["prominence"][ref]
                if gain > best:
                    best = float(gain)
                    detail = f"{line['freq']:.1f} Hz, {sensor.label.lower()}"
        measures["new_lines"] = {"value": best, "detail": detail}

        extra, worst_sensor = -np.inf, ""
        for sensor in SENSORS:
            delta = (
                res["rejection"][key][sensor.key]["calibrated_pct"]
                - res["rejection"][ref][sensor.key]["calibrated_pct"]
            )
            if delta > extra:
                extra, worst_sensor = float(delta), sensor.label
        measures["epoch_loss"] = {"value": extra, "detail": f"worst on {worst_sensor.lower()}"}

        for rule in VERDICT_RULES:
            m = measures[rule.key]
            m["severity"] = _severity(m["value"], rule.minor, rule.material)
            m["rule"] = rule

        overall = max(
            (m["severity"] for m in measures.values()),
            key=lambda sev: SEVERITY_ORDER.index(sev),
        )
        verdicts[key] = {
            "measures": measures,
            "overall": overall,
            "reference": ref,
            "reference_label": CONDITION_BY_KEY[ref].label,
        }
    return verdicts


def compute_all(
    dataset: Dataset | Path | str | None = None,
    params: Params = PARAMS,
    cache: Path | None = None,
) -> Results:
    if cache is not None and cache.exists():
        log.info("loading cached metrics from %s", cache)
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    t0 = time.time()
    if not isinstance(dataset, Dataset):
        dataset = load_dataset(dataset or (DATA_ROOT / DEFAULT_SITE))
    log.info("site %s, reading from %s", dataset.site, dataset.root)
    runs = load_all(dataset, params=params)

    line_freq = resolve_line_freq(runs, dataset)
    bands = bands_for(line_freq, fmax=params.psd_fmax)
    log.info("mains frequency %.0f Hz", line_freq)
    from .loading import _check_comparable

    warnings = _check_comparable(runs)

    log.info("spectra")
    psds = mx.compute_psds(runs, params)
    band_tbl = mx.band_table(psds, runs, bands=bands)

    log.info("time-domain amplitude")
    amplitudes = mx.amplitude_table(runs, params)

    log.info("sliding amplitude")
    sliding = {
        "analysis": mx.sliding_rms(runs, "mag", 1.0, 45.0, params),
        "drift": mx.sliding_rms(runs, "mag", 0.1, 1.0, params),
        "grad_analysis": mx.sliding_rms(runs, "grad", 1.0, 45.0, params),
    }

    log.info("spectral lines")
    lines = mx.detect_lines(psds, runs, params, line_freq=line_freq)
    source_distance = mx.source_distance_diagnostic(band_tbl, runs, bands=bands)

    log.info("epoch rejection")
    rejection = mx.calibrate_rejection(mx.epoch_rejection(runs, params))
    quality = mx.channel_quality(runs)

    log.info("event-locked analyses")
    button, fine, locked, tfr, motion = {}, {}, {}, {}, {}
    for key, run in runs.items():
        if not len(run.events):
            continue
        button[key] = ev.button_evoked(run, params)
        fine[key] = ev.fine_transient(run, params)
        locked[key] = ev.locked_response_test(run, params, line_freq=line_freq)
        t = ev.button_tfr(run, "mag", params)
        if t is not None:
            tfr[key] = t
    if "motion" in runs and len(runs["motion"].events):
        motion["motion"] = ev.motion_cycle(runs["motion"], params)

    ref = runs[BASELINE_KEY].raw
    mag_info = ref.copy().pick("mag").info
    topo_info = {"mag": mag_info, "grad": ref.copy().pick("grad").info}
    mag_names = ref.copy().pick("mag").ch_names

    res_dict = {
        "amplitudes": amplitudes,
        "lines": lines,
        "rejection": rejection,
    }
    verdicts = compute_verdicts(res_dict)

    results = Results(
        params=params,
        site=dataset.site,
        dataset_root=str(dataset.root),
        line_freq=line_freq,
        bands_def=bands,
        inventory=_inventory(runs),
        acquisition_warnings=warnings,
        psds=psds,
        bands=band_tbl,
        amplitudes=amplitudes,
        sliding=sliding,
        lines=lines,
        source_distance=source_distance,
        rejection=rejection,
        quality=quality,
        button=button,
        fine=fine,
        locked=locked,
        tfr=tfr,
        motion=motion,
        verdicts=verdicts,
        topo_info=topo_info,
        mag_names=mag_names,
        mne_version=mne.__version__,
        computed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_s=time.time() - t0,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(results, fh)
        log.info("cached metrics to %s", cache)
    return results
