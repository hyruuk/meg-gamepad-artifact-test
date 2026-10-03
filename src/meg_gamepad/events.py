"""Button-press-locked and motion-cycle-locked analyses.

These answer the question the broadband measures cannot: does an *event* on the
controller put a deterministic transient into the MEG? The hard part is that a
genuine neural/muscular response to pressing a button looks, at first glance,
exactly like an artifact. The analyses here are built to separate the two:

* a switching or trigger-crosstalk artifact is locked to the TTL edge within a
  couple of milliseconds, and its topography follows the hardware (focal at the
  sensors nearest the device, or near-uniform for a distant source);
* a motor/somatosensory response begins *before* switch closure (muscle activity
  precedes it), peaks tens to hundreds of milliseconds later, and has a dipolar
  topography over sensorimotor cortex.

A surrogate-event control gives the noise floor for both.
"""

from __future__ import annotations

import logging

import mne
import numpy as np

from .config import PARAMS, SENSOR_BY_KEY, SENSORS, Params
from .loading import Run

log = logging.getLogger(__name__)


def _epochs(
    run: Run,
    sensor_key: str,
    tmin: float,
    tmax: float,
    baseline: tuple[float, float] | None,
    events: np.ndarray | None = None,
) -> mne.Epochs | None:
    events = run.events if events is None else events
    if events is None or not len(events):
        return None
    return mne.Epochs(
        run.raw.copy().pick(sensor_key),
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None,
        proj=False,
        event_repeated="drop",
        verbose="ERROR",
    )


def button_evoked(run: Run, params: Params = PARAMS) -> dict:
    """Average response to button onsets, on the usual evoked timescale."""
    out: dict[str, dict] = {}
    for sensor in SENSORS:
        epochs = _epochs(
            run, sensor.key, params.ev_tmin, params.ev_tmax, params.ev_baseline
        )
        if epochs is None:
            continue
        evoked = epochs.average()
        data = evoked.data * sensor.scale
        times = evoked.times
        post = (times >= -0.05) & (times <= 0.3)
        pre = (times >= params.ev_baseline[0]) & (times <= params.ev_baseline[1])
        peak = np.abs(data[:, post]).max(axis=1)
        noise = data[:, pre].std(axis=1)
        worst = int(np.argmax(peak))
        # Latency must be measured in the same window the peak was taken from,
        # otherwise it can report an edge of the epoch that the peak never saw.
        post_idx = np.flatnonzero(post)
        peak_idx = int(post_idx[np.argmax(np.abs(data[worst, post]))])
        out[sensor.key] = {
            "times": times,
            "data": data,
            "ch_names": evoked.ch_names,
            "n_epochs": int(len(epochs)),
            "gfp": np.sqrt((data**2).mean(axis=0)),
            "peak_per_channel": peak,
            "prestim_sd": noise,
            "peak_max": float(peak.max()),
            "peak_median": float(np.median(peak)),
            "snr_max": float(np.max(peak / np.maximum(noise, 1e-12))),
            "worst_channel": evoked.ch_names[worst],
            "worst_latency_ms": float(times[peak_idx] * 1000),
            "unit": sensor.unit,
        }
    return out


def fine_transient(run: Run, params: Params = PARAMS) -> dict:
    """Millisecond-scale average around the TTL edge, with no extra filtering.

    An electrical artifact from the switch or the trigger line would appear here
    as a step or spike confined to roughly +/- 3 ms of t = 0 (the 330 Hz
    acquisition low-pass is what sets that smearing), and would be absent from
    the surrogate control.
    """
    out: dict[str, dict] = {}
    for sensor in SENSORS:
        epochs = _epochs(
            run, sensor.key, params.fine_tmin, params.fine_tmax, params.fine_baseline
        )
        if epochs is None:
            continue
        evoked = epochs.average()
        data = evoked.data * sensor.scale
        times = evoked.times
        edge = (times >= -0.002) & (times <= 0.005)
        pre = times <= params.fine_baseline[1]
        peak = np.abs(data[:, edge]).max(axis=1)
        noise = np.maximum(data[:, pre].std(axis=1), 1e-12)
        out[sensor.key] = {
            "times": times,
            "data": data,
            "ch_names": evoked.ch_names,
            "n_epochs": int(len(epochs)),
            "edge_peak_max": float(peak.max()),
            "edge_peak_median": float(np.median(peak)),
            "edge_z_max": float(np.max(peak / noise)),
            "worst_channel": evoked.ch_names[int(np.argmax(peak))],
            "unit": sensor.unit,
        }
    return out


def _locked_statistics(
    data_bb: np.ndarray,
    data_hp: np.ndarray,
    events: np.ndarray,
    first_samp: int,
    sfreq: float,
    params: Params,
    scale: float,
    line_freq: float,
) -> dict:
    """Three statistics of one event-locked average, computed by hand.

    Done with array slicing rather than mne.Epochs because this runs once per
    surrogate iteration and the Epochs machinery dominates the runtime.

    * ``peak`` -- largest deflection anywhere in the response window. Sensitive to
      anything locked to the events, including slow drift.
    * ``hp_peak`` -- the same on high-pass filtered data, which removes drift and
      leaves only genuinely fast responses.
    * ``line60`` -- amplitude of the mains-frequency component of the average.
      Non-zero only if the events are phase-locked to the mains supply.
    """
    n_pre = int(round(-params.ev_tmin * sfreq))
    n_post = int(round(params.ev_tmax * sfreq))
    n_len = n_pre + n_post + 1
    b0 = int(round((params.ev_baseline[0] - params.ev_tmin) * sfreq))
    b1 = int(round((params.ev_baseline[1] - params.ev_tmin) * sfreq))

    starts = events[:, 0] - first_samp - n_pre
    ok = (starts >= 0) & (starts + n_len <= data_bb.shape[1])
    starts = starts[ok]
    if len(starts) == 0:
        return {"peak": np.nan, "hp_peak": np.nan, "line60": np.nan, "n": 0}

    def average(data: np.ndarray) -> np.ndarray:
        acc = np.zeros((data.shape[0], n_len))
        for st in starts:
            seg = data[:, st : st + n_len]
            acc += seg - seg[:, b0:b1].mean(axis=1, keepdims=True)
        return acc / len(starts) * scale

    avg_bb = average(data_bb)
    avg_hp = average(data_hp)

    times = np.arange(n_len) / sfreq + params.ev_tmin
    post = (times >= -0.05) & (times <= 0.3)

    # Mains-frequency component of the broadband average, over the post window
    seg = avg_bb[:, post]
    t_seg = times[post]
    ref = np.exp(-2j * np.pi * line_freq * t_seg)
    line60 = float(np.mean(np.abs(2 * (seg * ref).mean(axis=1))))

    return {
        "peak": float(np.abs(avg_bb[:, post]).max()),
        "hp_peak": float(np.abs(avg_hp[:, post]).max()),
        "line60": line60,
        "n": int(len(starts)),
    }


def locked_response_test(
    run: Run, params: Params = PARAMS, line_freq: float = 60.0
) -> dict:
    """Test the button-locked average against a random-event null.

    The null is what the same number of epochs, averaged at random times in the
    same recording, produces. Because the statistic is a maximum over all
    channels and the whole response window, the comparison is already corrected
    for that search.
    """
    if not len(run.events):
        return {}
    rng = np.random.default_rng(params.random_seed)
    n_ev = len(run.events)
    lo = int(abs(params.ev_tmin) * run.sfreq) + 1
    hi = run.raw.n_times - int(params.ev_tmax * run.sfreq) - 1

    out: dict[str, dict] = {}
    for sensor in SENSORS:
        picked = run.raw.copy().pick(sensor.key)
        data_bb = picked.get_data()
        data_hp = (
            picked.copy()
            .filter(params.hp_evoked_hz, None, verbose="ERROR")
            .get_data()
        )
        common = dict(
            data_bb=data_bb, data_hp=data_hp, first_samp=picked.first_samp,
            sfreq=run.sfreq, params=params, scale=sensor.scale,
            line_freq=line_freq,
        )
        real = _locked_statistics(events=run.events, **common)

        null = {k: np.empty(params.n_surrogate) for k in ("peak", "hp_peak", "line60")}
        for i in range(params.n_surrogate):
            offs = np.sort(rng.choice(np.arange(lo, hi), size=n_ev, replace=False))
            fake = np.c_[offs + picked.first_samp, np.zeros(n_ev, int), np.ones(n_ev, int)]
            stat = _locked_statistics(events=fake, **common)
            for k in null:
                null[k][i] = stat[k]

        entry: dict = {"n_epochs": real["n"], "unit": sensor.unit}
        for k in ("peak", "hp_peak", "line60"):
            v, dist = real[k], null[k]
            entry[k] = {
                "real": v,
                "null": dist,
                "null_mean": float(dist.mean()),
                "null_p95": float(np.percentile(dist, 95)),
                "ratio": float(v / dist.mean()) if dist.mean() else float("nan"),
                "p_value": float((np.sum(dist >= v) + 1) / (len(dist) + 1)),
                "significant": bool(v > np.percentile(dist, 95)),
            }
        out[sensor.key] = entry
    return out


def button_tfr(run: Run, sensor_key: str = "mag", params: Params = PARAMS) -> dict | None:
    """Induced power around the press, as percent change from the epoch mean.

    A broadband vertical stripe at t = 0 is the artifact signature.

    Two details matter here. The wavelets are computed on a window far wider
    than the one displayed, because a Morlet wavelet that runs off the end of
    its epoch produces a large edge artifact; without the padding those
    artifacts fall inside the display window and dominate the map. And the
    reference is the mean of the whole epoch rather than a pre-stimulus
    baseline: presses in this run follow each other too closely for any
    pre-stimulus window to be free of neighbouring presses, so an epoch-mean
    reference is the only honest one. The question the map answers is therefore
    "is power concentrated at t = 0", which is exactly the artifact question.
    """
    epochs = _epochs(
        run, sensor_key, params.tfr_pad_tmin, params.tfr_pad_tmax, None
    )
    if epochs is None or len(epochs) < 5:
        return None
    freqs = np.logspace(
        np.log10(params.tfr_fmin), np.log10(params.tfr_fmax), params.tfr_n_freqs
    )
    n_cycles = np.clip(
        freqs / params.tfr_cycle_ratio, params.tfr_cycles_min, params.tfr_cycles_max
    )
    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        decim=params.tfr_decim,
        average=True,
        return_itc=False,
        verbose="ERROR",
    )
    # Trim the padding (and the edge artifacts living in it) BEFORE taking the
    # reference, so the epoch mean is computed over clean data only, then crop
    # again to the window actually displayed.
    tfr.crop(tmin=params.tfr_ref_tmin, tmax=params.tfr_ref_tmax)
    tfr.apply_baseline((None, None), mode="percent", verbose="ERROR")
    tfr.crop(tmin=params.ev_tmin, tmax=params.ev_tmax)
    return {
        "times": tfr.times,
        "freqs": freqs,
        "power_pct": tfr.data.mean(axis=0) * 100.0,
        "sensor": sensor_key,
        "n_epochs": int(len(epochs)),
        "pad": (params.tfr_pad_tmin, params.tfr_pad_tmax),
        "ref_window": (params.tfr_ref_tmin, params.tfr_ref_tmax),
        "n_cycles_range": (float(n_cycles.min()), float(n_cycles.max())),
    }


def motion_cycle(run: Run, params: Params = PARAMS) -> dict:
    """Low-frequency field locked to the controller reaching the face.

    In the motion run a button press marks the top of each cycle, which turns
    the movement into an event-locked measurement of the drift it causes.
    """
    out: dict[str, dict] = {}
    for sensor in SENSORS:
        if not len(run.events):
            continue
        picked = run.raw.copy().pick(sensor.key)
        picked.filter(None, params.motion_lowpass, verbose="ERROR")
        epochs = mne.Epochs(
            picked,
            run.events,
            tmin=params.motion_tmin,
            tmax=params.motion_tmax,
            baseline=None,
            preload=True,
            reject=None,
            proj=False,
            event_repeated="drop",
            verbose="ERROR",
        )
        if not len(epochs):
            continue
        data = epochs.get_data(copy=False) * sensor.scale
        evoked = data.mean(axis=0)
        # Peak-to-peak of the cycle-locked field, per channel.
        p2p = evoked.max(axis=1) - evoked.min(axis=1)
        out[sensor.key] = {
            "times": epochs.times,
            "evoked": evoked,
            "ch_names": epochs.ch_names,
            "n_epochs": int(len(epochs)),
            "p2p_per_channel": p2p,
            "p2p_median": float(np.median(p2p)),
            "p2p_max": float(p2p.max()),
            "worst_channel": epochs.ch_names[int(np.argmax(p2p))],
            "gfp": np.sqrt((evoked**2).mean(axis=0)),
            "unit": sensor.unit,
        }
    return out
