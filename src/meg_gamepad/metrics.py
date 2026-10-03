"""Quantitative artifact measures.

Every measure is expressed relative to the baseline run, because the absolute
noise level of a MEG room on a given afternoon is not the quantity of interest --
the *change* introduced by the controller is.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .config import (
    AMP_WINDOWS,
    BANDS,
    BASELINE_KEY,
    PARAMS,
    SENSORS,
    Params,
)
from .loading import Run, channel_site

log = logging.getLogger(__name__)

EPS = 1e-30


def db(ratio: np.ndarray | float) -> np.ndarray | float:
    """Power ratio -> decibels, guarded against zeros."""
    return 10.0 * np.log10(np.maximum(np.asarray(ratio, float), EPS))


def db_amp(ratio: np.ndarray | float) -> np.ndarray | float:
    """Amplitude (RMS) ratio -> decibels, so it is comparable with `db`."""
    return 20.0 * np.log10(np.maximum(np.asarray(ratio, float), EPS))


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------


def compute_psds(runs: dict[str, Run], params: Params = PARAMS) -> dict:
    """Welch PSD per run and sensor type, kept per channel.

    Returns ``{sensor_key: {"freqs": f, "psd": {run_key: (n_ch, n_freq)},
    "ch_names": [...]}}``.
    """
    out: dict[str, dict] = {}
    for sensor in SENSORS:
        psd_by_run: dict[str, np.ndarray] = {}
        freqs = None
        ch_names: list[str] = []
        for key, run in runs.items():
            picked = run.raw.copy().pick(sensor.key)
            spectrum = picked.compute_psd(
                method="welch",
                fmin=params.psd_fmin,
                fmax=params.psd_fmax,
                n_fft=params.psd_n_fft,
                n_overlap=params.psd_overlap,
                verbose="ERROR",
            )
            psd, freqs = spectrum.get_data(return_freqs=True)
            # to (unit)^2 / Hz
            psd_by_run[key] = psd * sensor.scale**2
            ch_names = picked.ch_names
        out[sensor.key] = {"freqs": freqs, "psd": psd_by_run, "ch_names": ch_names}
    return out


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Mean PSD within a band, per channel."""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        return np.full(psd.shape[0], np.nan)
    return psd[:, mask].mean(axis=1)


def band_table(psds: dict, runs: dict[str, Run], bands: tuple = BANDS) -> dict:
    """Per-band dB change vs baseline, per sensor type, per channel + summary."""
    table: dict[str, dict] = {}
    for sensor in SENSORS:
        entry = psds[sensor.key]
        freqs, psd = entry["freqs"], entry["psd"]
        per_band: dict[str, dict] = {}
        for band in bands:
            base = band_power(psd[BASELINE_KEY], freqs, band.fmin, band.fmax)
            per_run = {}
            for key in runs:
                cur = band_power(psd[key], freqs, band.fmin, band.fmax)
                change = db(cur / np.maximum(base, EPS))
                per_run[key] = {
                    "abs_amp": float(np.median(np.sqrt(cur))),  # unit/sqrt(Hz)
                    "db_median": float(np.median(change)),
                    "db_p95": float(np.percentile(change, 95)),
                    "db_max": float(np.max(change)),
                    "db_per_channel": change,
                }
            per_band[band.key] = per_run
        table[sensor.key] = per_band
    return table


# --------------------------------------------------------------------------
# Time-domain amplitude
# --------------------------------------------------------------------------


def amplitude_table(runs: dict[str, Run], params: Params = PARAMS) -> dict:
    """RMS amplitude per channel in each passband, plus the as-recorded RMS.

    The as-recorded ("broadband") figure is reported separately because on
    magnetometers it is dominated by sub-1 Hz environmental drift, which swamps
    everything else and would otherwise hide the comparison.
    """
    out: dict[str, dict] = {}
    for sensor in SENSORS:
        per_window: dict[str, dict] = {}
        broadband: dict[str, np.ndarray] = {}
        for key, run in runs.items():
            picked = run.raw.copy().pick(sensor.key)
            data = picked.get_data() * sensor.scale
            broadband[key] = data.std(axis=1)
        per_window["broadband"] = broadband
        for window in AMP_WINDOWS:
            vals: dict[str, np.ndarray] = {}
            for key, run in runs.items():
                picked = run.raw.copy().pick(sensor.key)
                picked.filter(
                    window.fmin,
                    min(window.fmax, run.lowpass),
                    verbose="ERROR",
                    h_trans_bandwidth="auto",
                    l_trans_bandwidth="auto",
                )
                vals[key] = picked.get_data().std(axis=1) * sensor.scale
            per_window[window.key] = vals
        out[sensor.key] = per_window
    return out


def sliding_rms(
    runs: dict[str, Run], sensor_key: str, fmin: float, fmax: float, params: Params = PARAMS
) -> dict:
    """Median-across-channels RMS in a sliding window -- shows burst structure."""
    from .config import SENSOR_BY_KEY

    sensor = SENSOR_BY_KEY[sensor_key]
    out: dict[str, dict] = {}
    for key, run in runs.items():
        picked = run.raw.copy().pick(sensor_key)
        picked.filter(fmin, min(fmax, run.lowpass), verbose="ERROR")
        data = picked.get_data() * sensor.scale
        n_win = int(params.win_sec * run.sfreq)
        step = int(params.win_step_sec * run.sfreq)
        starts = np.arange(0, data.shape[1] - n_win, step)
        rms = np.empty((data.shape[0], len(starts)))
        for i, s in enumerate(starts):
            seg = data[:, s : s + n_win]
            rms[:, i] = seg.std(axis=1)
        out[key] = {
            "times": starts / run.sfreq + params.win_sec / 2,
            "median": np.median(rms, axis=0),
            "p95": np.percentile(rms, 95, axis=0),
            "max": rms.max(axis=0),
        }
    return out


# --------------------------------------------------------------------------
# Narrowband spectral lines
# --------------------------------------------------------------------------


def _prominence_spectrum(
    median_psd: np.ndarray, freqs: np.ndarray, params: Params
) -> np.ndarray:
    """dB by which each bin exceeds the local smooth spectral floor."""
    df = float(np.median(np.diff(freqs)))
    size = max(3, int(round(2 * params.line_floor_hz / df)) | 1)
    floor = median_filter(median_psd, size=size, mode="nearest")
    return db(median_psd / np.maximum(floor, EPS))


def detect_lines(
    psds: dict, runs: dict[str, Run], params: Params = PARAMS,
    line_freq: float | None = None,
) -> dict:
    """Find narrowband lines and compare their prominence across runs.

    A line that is equally prominent in the baseline is an existing feature of
    the environment; one that grows only when the controller is present is the
    signature we are hunting. Powerline harmonics are tagged separately because
    they are expected and are not attributable to the device without more
    evidence.
    """
    line_freq = params.powerline_hz if line_freq is None else float(line_freq)
    results: dict[str, dict] = {}
    for sensor in SENSORS:
        entry = psds[sensor.key]
        freqs, psd = entry["freqs"], entry["psd"]
        prom = {
            key: _prominence_spectrum(np.median(psd[key], axis=0), freqs, params)
            for key in runs
        }
        df = float(np.median(np.diff(freqs)))
        min_dist = max(1, int(round(1.0 / df)))

        # Candidate frequencies: any peak in any gamepad-on run.
        candidates: set[int] = set()
        for key, run in runs.items():
            if not run.condition.gamepad_on:
                continue
            idx, _ = find_peaks(
                prom[key], height=params.line_min_prominence_db, distance=min_dist
            )
            candidates.update(idx.tolist())

        lines = []
        for i in sorted(candidates):
            f0 = float(freqs[i])
            base_prom = float(prom[BASELINE_KEY][i])
            per_run = {key: float(prom[key][i]) for key in runs}
            gain = {
                key: per_run[key] - base_prom
                for key, run in runs.items()
                if run.condition.gamepad_on
            }
            best_key = max(gain, key=gain.get)
            harmonic = round(f0 / line_freq)
            is_powerline = harmonic >= 1 and abs(f0 - harmonic * line_freq) <= (
                params.powerline_tol_hz
            )
            if is_powerline:
                klass = "powerline"
            elif f0 < params.line_verdict_fmin:
                klass = "low-frequency"
            else:
                klass = "instrument candidate"
            lines.append(
                {
                    "klass": klass,
                    "freq": f0,
                    "prominence": per_run,
                    "baseline_prominence": base_prom,
                    "gain": gain,
                    "max_gain": float(gain[best_key]),
                    "max_gain_run": best_key,
                    "is_powerline": bool(is_powerline),
                    "harmonic": int(harmonic) if is_powerline else None,
                    "is_new": bool(gain[best_key] >= params.line_excess_db),
                    #: only an above-threshold, non-powerline peak in the band
                    #: where instrument lines actually live counts for the verdict
                    "counts_for_verdict": bool(
                        gain[best_key] >= params.line_excess_db
                        and klass == "instrument candidate"
                    ),
                }
            )
        lines.sort(key=lambda d: -d["max_gain"])
        results[sensor.key] = {"lines": lines, "prominence": prom, "freqs": freqs}
    return results


def excess_crossover(
    psds: dict, sensor_key: str, cond_key: str, ref_key: str, db_threshold: float = 3.0
) -> float | None:
    """Highest frequency at which `cond` still exceeds `ref` by `db_threshold`.

    Turns "the effect is low-frequency" into a number, so a band-average that
    looks alarming can be placed correctly: an excess confined to 1-2 Hz inside
    a 1-45 Hz band is a very different problem from one spread across it.
    """
    from scipy.ndimage import uniform_filter1d

    entry = psds[sensor_key]
    freqs = entry["freqs"]
    ratio = db(
        np.median(entry["psd"][cond_key], axis=0)
        / np.maximum(np.median(entry["psd"][ref_key], axis=0), EPS)
    )
    smooth = uniform_filter1d(ratio, 21)
    above = np.flatnonzero(smooth > db_threshold)
    return float(freqs[above[-1]]) if len(above) else None


def source_distance_diagnostic(
    band_tbl: dict, runs: dict[str, Run], bands: tuple = BANDS
) -> dict:  # noqa: D401
    """Magnetometer-vs-gradiometer excess, as a crude source-distance cue.

    Magnetometers respond to near-uniform fields from distant sources; planar
    gradiometers suppress those but stay sensitive to nearby sources. So excess
    that shows up on magnetometers *only* points away from the participant's
    hands and towards the environment, while excess shared by both points at
    something close to the helmet.
    """
    out: dict[str, dict] = {}
    for key, run in runs.items():
        if not run.condition.gamepad_on:
            continue
        per_band = {}
        for band in bands:
            mag = band_tbl["mag"][band.key][key]["db_median"]
            grad = band_tbl["grad"][band.key][key]["db_median"]
            per_band[band.key] = {
                "mag_db": mag,
                "grad_db": grad,
                "mag_minus_grad_db": mag - grad,
            }
        out[key] = per_band
    return out


# --------------------------------------------------------------------------
# Practical cost: epoch rejection
# --------------------------------------------------------------------------


def epoch_rejection(runs: dict[str, Run], params: Params = PARAMS) -> dict:
    """Fraction of fixed-length epochs a standard peak-to-peak criterion kills.

    This converts the spectral story into the number an experimenter actually
    cares about: how much data would I lose?
    """
    out: dict[str, dict] = {}
    lo, hi = params.reject_band
    for key, run in runs.items():
        per_sensor = {}
        for sensor in SENSORS:
            picked = run.raw.copy().pick(sensor.key)
            # Filter first. On unfiltered magnetometers a peak-to-peak criterion
            # mostly measures sub-1 Hz environmental drift, which every real
            # pipeline removes before it ever rejects an epoch.
            picked.filter(lo, min(hi, run.lowpass), verbose="ERROR")
            events = mne.make_fixed_length_events(
                picked, duration=params.epoch_sec, first_samp=True
            )
            epochs = mne.Epochs(
                picked,
                events,
                tmin=0.0,
                tmax=params.epoch_sec,
                baseline=None,
                preload=True,
                reject=None,
                verbose="ERROR",
            )
            data = epochs.get_data(copy=False) * sensor.scale
            p2p = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_ch)
            worst = p2p.max(axis=1)
            bad = worst > sensor.reject_p2p
            # A single threshold is an arbitrary cut, and at conventional values
            # clean data rejects nothing at all, so the whole curve is kept and
            # the report plots it.
            sweep = sensor.reject_p2p * np.logspace(-1.2, 0.5, 40)
            curve = [float(100.0 * (worst > th).mean()) for th in sweep]
            per_sensor[sensor.key] = {
                "n_epochs": int(len(epochs)),
                "n_rejected": int(bad.sum()),
                "pct_rejected": float(100.0 * bad.mean()),
                "median_p2p": float(np.median(p2p)),
                "p95_p2p": float(np.percentile(p2p, 95)),
                "worst_p2p_per_epoch": worst,
                "sweep_thresholds": sweep,
                "sweep_pct": np.array(curve),
                "threshold": sensor.reject_p2p,
            }
        out[key] = per_sensor
    return out


def calibrate_rejection(rejection: dict, baseline_key: str = BASELINE_KEY,
                        target_pct: float = 5.0) -> dict:
    """Add a threshold calibrated so the baseline run loses `target_pct` of epochs.

    The conventional 4000 fT / 4000 fT/cm criterion rejects nothing at all once
    the data are band-passed, so it cannot discriminate between conditions. Fixing
    the criterion on the reference run instead gives every condition the same
    yardstick and lets the comparison say something.
    """
    for sensor in SENSORS:
        base = rejection[baseline_key][sensor.key]["worst_p2p_per_epoch"]
        thresh = float(np.percentile(base, 100.0 - target_pct))
        for key in rejection:
            worst = rejection[key][sensor.key]["worst_p2p_per_epoch"]
            rejection[key][sensor.key]["calibrated_threshold"] = thresh
            rejection[key][sensor.key]["calibrated_pct"] = float(
                100.0 * (worst > thresh).mean()
            )
            rejection[key][sensor.key]["calibrated_target"] = target_pct
    return rejection


def channel_quality(runs: dict[str, Run]) -> dict:
    """MNE's own flat/noisy channel detection, run per condition.

    A channel that only goes bad when the controller is present would be a
    strong localised-artifact signal.
    """
    out: dict[str, dict] = {}
    for key, run in runs.items():
        picked = run.raw.copy().pick(["mag", "grad"])
        try:
            _, flat = mne.preprocessing.annotate_amplitude(
                picked, flat=0.0, bad_percent=5.0, verbose="ERROR"
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            log.warning("annotate_amplitude failed for %s: %s", key, exc)
            flat = []
        out[key] = {"flat": list(flat), "header_bads": run.header_bads}
    return out


# --------------------------------------------------------------------------
# Per-site topography helper
# --------------------------------------------------------------------------


def collapse_grad_pairs(values: np.ndarray, ch_names: list[str], mag_names: list[str]) -> np.ndarray:
    """Average each planar-gradiometer pair onto its shared helmet site.

    Returned in the order of ``mag_names`` so the result can be plotted on the
    magnetometer layout.
    """
    by_site: dict[str, list[float]] = {}
    for name, val in zip(ch_names, values):
        by_site.setdefault(channel_site(name), []).append(float(val))
    return np.array(
        [np.mean(by_site[channel_site(m)]) for m in mag_names], dtype=float
    )
