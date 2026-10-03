"""Figures, rendered to base64 PNG so the report is a single portable file.

Styling follows one consistent system: the baseline is drawn in neutral ink
because it is a reference rather than a peer series; the three controller
conditions take the first three categorical slots (blue / orange / aqua), which
are the CVD-validated all-pairs subset. Grid and axes are recessive, marks are
thin, and every multi-series panel carries a legend so identity is never
colour-alone.
"""

from __future__ import annotations

import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .config import (
    AMP_WINDOWS,
    BANDS,
    BASELINE_KEY,
    CONDITION_BY_KEY,
    CONDITIONS,
    NON_BASELINE,
    PALETTE,
    SENSOR_BY_KEY,
    SENSORS,
)
from .metrics import collapse_grad_pairs, db, db_amp, EPS

log = logging.getLogger(__name__)

#: Concrete family names only -- matplotlib cannot resolve CSS keywords like
#: system-ui, and silently falls back when a weight is missing.
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

#: blue <-> red diverging ramp with a neutral grey midpoint (never a hue at zero)
DIVERGING = LinearSegmentedColormap.from_list(
    "meg_div",
    [PALETTE["diverging_neg"], "#9ec5f4", PALETTE["diverging_mid"], "#f0a3a3", PALETTE["diverging_pos"]],
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "font.size": 9,
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["ink_secondary"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelcolor": PALETTE["ink_secondary"],
            "ytick.labelcolor": PALETTE["ink_secondary"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": PALETTE["surface"],
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mark_harmonics(ax, fmax: float, line_freq: float = 60.0) -> None:
    n = 1
    while n * line_freq <= fmax:
        ax.axvline(
            n * line_freq, color=PALETTE["axis"], lw=0.6, ls=":", zorder=0
        )
        n += 1


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------


def fig_psd(res) -> str:
    """Absolute PSD per run, one panel per sensor type."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, sensor in zip(axes, SENSORS):
        entry = res.psds[sensor.key]
        freqs = entry["freqs"]
        for cond in CONDITIONS:
            med = np.median(entry["psd"][cond.key], axis=0)
            ax.loglog(
                freqs,
                np.sqrt(med),
                color=cond.color,
                label=cond.label,
                lw=1.8 if cond.key == BASELINE_KEY else 1.4,
                ls="--" if cond.key == BASELINE_KEY else "-",
                alpha=0.95,
            )
        _mark_harmonics(ax, freqs.max(), res.line_freq)
        ax.set_title(f"{sensor.label} — amplitude spectral density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"{sensor.unit}/√Hz")
        ax.set_xlim(freqs.min(), freqs.max())
        _despine(ax)
    axes[0].legend(loc="lower left", ncols=2)
    fig.text(
        0.5,
        -0.04,
        "The curves give the median across channels. The dotted vertical lines "
        f"mark the {res.line_freq:g} Hz powerline harmonics.",
        ha="center",
        color=PALETTE["muted"],
        fontsize=8,
    )
    return to_b64(fig)


def fig_psd_ratio(res) -> str:
    """PSD change vs baseline in dB -- the single most diagnostic panel."""
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, sensor in zip(axes, SENSORS):
        entry = res.psds[sensor.key]
        freqs = entry["freqs"]
        base = np.median(entry["psd"][BASELINE_KEY], axis=0)
        ax.axhspan(-3, 3, color=PALETTE["grid"], alpha=0.55, zorder=0, lw=0)
        ax.axhline(0, color=PALETTE["axis"], lw=0.9, zorder=1)
        for cond in NON_BASELINE:
            med = np.median(entry["psd"][cond.key], axis=0)
            ax.semilogx(
                freqs, db(med / np.maximum(base, EPS)), color=cond.color,
                label=cond.label, lw=1.3,
            )
        _mark_harmonics(ax, freqs.max(), res.line_freq)
        ax.set_title(f"{sensor.label} — change vs baseline")
        ax.set_ylabel("dB re baseline")
        ax.set_xlim(freqs.min(), freqs.max())
        # Cover the full excursion of every curve: the low-frequency movement
        # effect and the mains harmonics are both findings, not clutter to clip.
        allv = np.concatenate(
            [
                db(np.median(entry["psd"][c.key], axis=0) / np.maximum(base, EPS))
                for c in NON_BASELINE
            ]
        )
        span = max(6.0, float(np.abs(allv).max()))
        ax.set_ylim(-span * 1.08, span * 1.08)
        _despine(ax)
    axes[-1].set_xlabel("Frequency (Hz)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncols=3,
        bbox_to_anchor=(0.5, 1.02), fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.text(
        0.5, 0.005,
        "The shaded band is +/-3 dB, which is the run-to-run variability of a "
        "quiet room. A positive value means more noise with the controller. The "
        "deep notch near 10 Hz is alpha desynchronisation from the participant.",
        ha="center", color=PALETTE["muted"], fontsize=8,
    )
    return to_b64(fig)


def fig_amplitude_distributions(res) -> str:
    """Per-channel RMS distributions, small-multiplied by passband."""
    apply_style()
    windows = [("broadband", "As recorded (0.1–330 Hz)")] + [
        (w.key, w.label) for w in AMP_WINDOWS
    ]
    fig, axes = plt.subplots(
        2, len(windows), figsize=(3.0 * len(windows), 6.0), squeeze=False
    )
    for row, sensor in enumerate(SENSORS):
        for col, (wkey, wlabel) in enumerate(windows):
            ax = axes[row][col]
            vals = res.amplitudes[sensor.key][wkey]
            data = [vals[c.key] for c in CONDITIONS]
            bp = ax.boxplot(
                data,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color=PALETTE["ink"], lw=1.4),
                whiskerprops=dict(color=PALETTE["axis"], lw=0.9),
                capprops=dict(color=PALETTE["axis"], lw=0.9),
            )
            for patch, cond in zip(bp["boxes"], CONDITIONS):
                patch.set_facecolor(cond.color)
                patch.set_alpha(0.30)
                patch.set_edgecolor(cond.color)
                patch.set_linewidth(1.3)
            ax.set_xticks(range(1, len(CONDITIONS) + 1))
            ax.set_xticklabels(
                [c.label.replace(" ", "\n") for c in CONDITIONS], fontsize=7.5
            )
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(f"{sensor.label}\nRMS ({sensor.unit})")
            ax.set_title(wlabel, fontsize=9)
            _despine(ax)
    fig.suptitle(
        "Per-channel RMS amplitude by passband", y=1.01, fontsize=11, color=PALETTE["ink"]
    )
    fig.text(
        0.5, -0.03,
        "The box gives the interquartile range across channels. The line gives "
        "the median. The vertical axis is logarithmic. Sub-1 Hz environmental "
        "drift dominates the as-recorded panel.",
        ha="center", color=PALETTE["muted"], fontsize=8,
    )
    fig.tight_layout()
    return to_b64(fig)


def fig_sliding(res) -> str:
    """Amplitude over time -- is the artifact continuous, or bursty?"""
    apply_style()
    panels = [
        ("drift", "Magnetometers, 0.1–1 Hz (drift / movement)"),
        ("analysis", "Magnetometers, 1–45 Hz (analysis band)"),
        ("grad_analysis", "Gradiometers, 1–45 Hz (analysis band)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 7.2), sharex=True)
    for ax, (pkey, title) in zip(axes, panels):
        unit = "fT/cm" if pkey.startswith("grad") else "fT"
        for cond in CONDITIONS:
            d = res.sliding[pkey][cond.key]
            ax.plot(
                d["times"], d["median"], color=cond.color, label=cond.label,
                lw=1.5 if cond.key == BASELINE_KEY else 1.3,
                ls="--" if cond.key == BASELINE_KEY else "-",
            )
        ax.set_title(title)
        ax.set_ylabel(f"RMS ({unit})")
        _despine(ax)
    axes[-1].set_xlabel("Time within run (s)")
    axes[0].legend(loc="upper right", ncols=4)
    fig.text(
        0.5, 0.03,
        "The analysis uses 1 s windows with a step of 0.25 s. Each curve gives "
        "the median across channels.",
        ha="center", color=PALETTE["muted"], fontsize=8,
    )
    fig.tight_layout()
    return to_b64(fig)


# --------------------------------------------------------------------------
# Narrowband lines
# --------------------------------------------------------------------------


def fig_lines(res, sensor_key: str = "mag", top_n: int = 12) -> str:
    """Prominence gain of the strongest candidate lines, grouped by condition."""
    apply_style()
    lines = res.lines[sensor_key]["lines"][:top_n]
    if not lines:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No narrowband lines detected", ha="center",
                color=PALETTE["muted"], transform=ax.transAxes)
        ax.axis("off")
        return to_b64(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_cond = len(NON_BASELINE)
    y = np.arange(len(lines))
    height = 0.8 / n_cond
    for i, cond in enumerate(NON_BASELINE):
        vals = [ln["gain"].get(cond.key, 0.0) for ln in lines]
        ax.barh(
            y + (i - (n_cond - 1) / 2) * height, vals, height=height * 0.88,
            color=cond.color, label=cond.label, edgecolor=PALETTE["surface"], linewidth=0.8,
        )
    ax.axvline(0, color=PALETTE["axis"], lw=0.9)
    ax.axvline(
        res.params.line_excess_db, color=PALETTE["ink_secondary"], lw=0.9, ls=":",
    )
    ax.text(
        res.params.line_excess_db, -0.75,
        f" {res.params.line_excess_db:g} dB threshold", fontsize=7.5,
        color=PALETTE["ink_secondary"], va="center", ha="left",
    )
    suffix = {
        "powerline": lambda ln: f"  ({res.line_freq:g}×{ln['harmonic']})",
        "low-frequency": lambda ln: "  (phys. band)",
        "instrument candidate": lambda ln: "  <- candidate",
    }
    labels = [f"{ln['freq']:.1f} Hz" + suffix[ln["klass"]](ln) for ln in lines]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain in peak prominence vs baseline (dB)")
    ax.set_title(
        f"{SENSOR_BY_KEY[sensor_key].label} — narrowband lines, "
        "prominence above the local spectral floor",
    )
    # Legend below the axes: the bars fill the plotting area at every row, so
    # there is no interior space for it.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=8.5,
    )
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.text(
        0.5, -0.17,
        f"({res.line_freq:g}xn) identifies a powerline harmonic. (phys. band) "
        "identifies a peak below 30 Hz. The alpha, mu and beta rhythms are narrowband in that range, "
        "so a gain there gives no information about the hardware. Only the rows "
        "marked '<- candidate' are evidence against the device.",
        ha="center", color=PALETTE["muted"], fontsize=8,
    )
    return to_b64(fig)


# --------------------------------------------------------------------------
# Topography
# --------------------------------------------------------------------------


def _topo_values(res, sensor_key: str, band_key: str, cond_key: str) -> np.ndarray:
    change = res.bands[sensor_key][band_key][cond_key]["db_per_channel"]
    if sensor_key == "grad":
        return collapse_grad_pairs(
            change, res.psds["grad"]["ch_names"], res.mag_names
        )
    return change


def fig_topography(res, cond_key: str, band_keys: tuple[str, ...] | None = None) -> str:
    """Where on the helmet the excess power sits, per band and sensor type.

    Global, smooth patterns point to a distant environmental source; focal
    patterns point to something close to the participant.
    """
    apply_style()
    bands = [b for b in res.bands_def if band_keys is None or b.key in band_keys]
    info = res.topo_info["mag"]
    fig, axes = plt.subplots(
        2, len(bands), figsize=(1.75 * len(bands) + 1.2, 4.6), squeeze=False
    )
    for row, sensor in enumerate(SENSORS):
        vals = [_topo_values(res, sensor.key, b.key, cond_key) for b in bands]
        vmax = max(2.0, float(np.percentile(np.abs(np.concatenate(vals)), 98)))
        for col, (band, v) in enumerate(zip(bands, vals)):
            ax = axes[row][col]
            im, _ = mne.viz.plot_topomap(
                v, info, axes=ax, show=False, cmap=DIVERGING,
                vlim=(-vmax, vmax), sensors=False, contours=0,
                outlines="head", extrapolate="head",
            )
            if row == 0:
                ax.set_title(band.label.split(" (")[0], fontsize=8.5, pad=4)
            if col == 0:
                ax.text(
                    -0.25, 0.5, sensor.label, rotation=90, va="center", ha="center",
                    transform=ax.transAxes, fontsize=8.5, color=PALETTE["ink_secondary"],
                )
        cbar = fig.colorbar(
            im, ax=axes[row].tolist(), fraction=0.025, pad=0.02, aspect=12
        )
        cbar.set_label("dB re baseline", fontsize=7.5, color=PALETTE["ink_secondary"])
        cbar.ax.tick_params(labelsize=7, colors=PALETTE["muted"])
        cbar.outline.set_visible(False)
    fig.suptitle(
        f"{CONDITION_BY_KEY[cond_key].label} — spatial distribution of power change",
        fontsize=10.5, y=1.02, color=PALETTE["ink"],
    )
    return to_b64(fig)


# --------------------------------------------------------------------------
# Data loss
# --------------------------------------------------------------------------


def fig_rejection(res) -> str:
    """Data loss as a function of the rejection threshold, not at one arbitrary cut."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, sensor in zip(axes, SENSORS):
        for cond in CONDITIONS:
            r = res.rejection[cond.key][sensor.key]
            ax.semilogx(
                r["sweep_thresholds"], r["sweep_pct"], color=cond.color,
                label=cond.label, lw=1.8 if cond.key == BASELINE_KEY else 1.4,
                ls="--" if cond.key == BASELINE_KEY else "-",
            )
        r0 = res.rejection[BASELINE_KEY][sensor.key]
        ax.axvline(r0["calibrated_threshold"], color=PALETTE["ink_secondary"], lw=1.0, ls=":")
        ax.annotate(
            f"calibrated\n{r0['calibrated_threshold']:,.0f} {sensor.unit}",
            xy=(r0["calibrated_threshold"], 72), fontsize=7,
            color=PALETTE["ink_secondary"], ha="right",
            xytext=(-5, 0), textcoords="offset points",
        )
        ax.axvline(sensor.reject_p2p, color=PALETTE["axis"], lw=1.0, ls="-.")
        ax.annotate(
            f"conventional\n{sensor.reject_p2p:,.0f} {sensor.unit}",
            xy=(sensor.reject_p2p, 72), fontsize=7, color=PALETTE["muted"],
            xytext=(5, 0), textcoords="offset points",
        )
        ax.set_xlabel(f"Peak-to-peak rejection threshold ({sensor.unit})")
        ax.set_ylabel("Epochs rejected (%)")
        ax.set_title(f"{sensor.label}")
        ax.set_ylim(-3, 103)
        _despine(ax)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Data loss across the whole range of rejection thresholds "
        f"({res.params.epoch_sec:g} s epochs, "
        f"{res.params.reject_band[0]:g}–{res.params.reject_band[1]:g} Hz)",
        fontsize=10.5, y=1.04, color=PALETTE["ink"],
    )
    fig.tight_layout()
    return to_b64(fig)


# --------------------------------------------------------------------------
# Button-locked
# --------------------------------------------------------------------------


def fig_button_evoked(res, cond_key: str) -> str:
    """Butterfly plot plus field power for the button-locked average."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    cond = CONDITION_BY_KEY[cond_key]
    for col, sensor in enumerate(SENSORS):
        b = res.button[cond_key][sensor.key]
        t = b["times"] * 1000
        ax = axes[0][col]
        ax.plot(t, b["data"].T, color=PALETTE["ink_secondary"], lw=0.4, alpha=0.30)
        worst = b["ch_names"].index(b["worst_channel"])
        ax.plot(t, b["data"][worst], color=cond.color, lw=1.8,
                label=f"{b['worst_channel']} (largest)")
        ax.axvline(0, color=PALETTE["ink"], lw=0.9)
        ax.set_ylabel(f"{sensor.unit}")
        ax.set_title(f"{sensor.label} — {b['n_epochs']} button presses")
        ax.legend(loc="upper left")
        _despine(ax)

        ax2 = axes[1][col]
        ax2.plot(t, b["gfp"], color=cond.color, lw=1.6)
        ax2.fill_between(t, 0, b["gfp"], color=cond.color, alpha=0.18, lw=0)
        ax2.axvline(0, color=PALETTE["ink"], lw=0.9)
        ax2.set_ylabel(f"Field power\n(RMS, {sensor.unit})")
        ax2.set_xlabel("Time from button TTL (ms)")
        _despine(ax2)
    fig.suptitle(
        f"{cond.label} — button-press-locked average", fontsize=11, y=1.0,
        color=PALETTE["ink"],
    )
    fig.tight_layout()
    return to_b64(fig)


def fig_fine_transient(res, cond_key: str) -> str:
    """Millisecond zoom on the TTL edge: is there a switching transient?"""
    apply_style()
    cond = CONDITION_BY_KEY[cond_key]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, sensor in zip(axes, SENSORS):
        f = res.fine[cond_key][sensor.key]
        t = f["times"] * 1000
        ax.plot(t, f["data"].T, color=PALETTE["ink_secondary"], lw=0.4, alpha=0.28)
        worst = f["ch_names"].index(f["worst_channel"])
        ax.plot(t, f["data"][worst], color=cond.color, lw=1.8,
                label=f"{f['worst_channel']} (largest)")
        ax.axvspan(-2, 5, color=PALETTE["warning"], alpha=0.12, lw=0,
                   label="edge window")
        ax.axvline(0, color=PALETTE["ink"], lw=1.0)
        ax.set_title(f"{sensor.label} — TTL edge, no extra filtering")
        ax.set_xlabel("Time from button TTL (ms)")
        ax.set_ylabel(sensor.unit)
        ax.legend(loc="upper left", fontsize=8)
        _despine(ax)
    fig.text(
        0.5, -0.07,
        "A switch artifact or a trigger-line artifact appears as a step or a "
        "spike in the shaded window only. The 330 Hz acquisition low-pass filter "
        "spreads a true step over approximately +/-3 ms.",
        ha="center", color=PALETTE["muted"], fontsize=8,
    )
    fig.tight_layout()
    return to_b64(fig)


STAT_LABEL = {
    "peak": ("Any locked deflection", "largest deflection in the average"),
    "hp_peak": ("Survives high-pass", "the same, after removal of the drift band"),
    "line60": ("Mains phase-locking", "mains-frequency component of the average"),
}


def fig_locked_tests(res, cond_key: str) -> str:
    """Each locked statistic against its random-event null."""
    apply_style()
    cond = CONDITION_BY_KEY[cond_key]
    stats = ("peak", "hp_peak", "line60")
    fig, axes = plt.subplots(2, len(stats), figsize=(11, 5.4))
    for row, sensor in enumerate(SENSORS):
        entry = res.locked[cond_key][sensor.key]
        for col, stat in enumerate(stats):
            d = entry[stat]
            ax = axes[row][col]
            ax.hist(
                d["null"], bins=26, color=PALETTE["ink_secondary"], alpha=0.28,
                edgecolor=PALETTE["surface"], linewidth=0.7, label="random-event null",
            )
            ax.axvline(d["null_p95"], color=PALETTE["ink_secondary"], ls=":", lw=1.2,
                       label="null 95th pct")
            ax.axvline(d["real"], color=cond.color, lw=2.2, label="observed")
            verdict = "above chance" if d["significant"] else "within chance"
            ax.set_title(
                f"{STAT_LABEL[stat][0]}\n{d['ratio']:.2f}× null · p = {d['p_value']:.3f} · {verdict}",
                fontsize=8.5,
            )
            ax.set_xlabel(f"{STAT_LABEL[stat][1]} ({sensor.unit})", fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"{sensor.label}\nsurrogate count", fontsize=8.5)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)
            _despine(ax)
    fig.suptitle(
        f"{cond.label} — test for a signal locked to the button press",
        fontsize=11, y=1.02, color=PALETTE["ink"],
    )
    fig.tight_layout()
    return to_b64(fig)


def fig_button_topography(res, cond_key: str) -> str:
    """Topography of the locked average at its peak -- dipolar, or hardware-shaped?"""
    apply_style()
    info = res.topo_info["mag"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    cond = CONDITION_BY_KEY[cond_key]
    for col, sensor in enumerate(SENSORS):
        b = res.button[cond_key][sensor.key]
        t = b["times"]
        # peak of field power inside the response window
        win = (t >= -0.05) & (t <= 0.3)
        tpk = t[win][int(np.argmax(b["gfp"][win]))]
        for j, (when, lat) in enumerate((("edge", 0.0), ("peak", float(tpk)))):
            ax = axes[col * 2 + j]
            idx = int(np.argmin(np.abs(t - lat)))
            v = b["data"][:, idx]
            if sensor.key == "grad":
                v = collapse_grad_pairs(v, b["ch_names"], res.mag_names)
            vmax = float(np.percentile(np.abs(v), 99)) or 1.0
            im, _ = mne.viz.plot_topomap(
                v, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-vmax, vmax),
                sensors=False, contours=6, outlines="head", extrapolate="head",
            )
            ax.set_title(
                f"{sensor.label.split()[0]}\n{when}: {lat*1000:+.0f} ms", fontsize=8.5
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            cbar.ax.tick_params(labelsize=6.5, colors=PALETTE["muted"])
            cbar.set_label(sensor.unit, fontsize=7, color=PALETTE["ink_secondary"])
            cbar.outline.set_visible(False)
    fig.suptitle(
        f"{cond.label} — field topography at the TTL edge and at the peak",
        fontsize=10.5, y=1.06, color=PALETTE["ink"],
    )
    fig.tight_layout()
    return to_b64(fig)


def fig_tfr(res, cond_key: str) -> str:
    """Induced power: a vertical broadband stripe is artifact, beta drop is motor."""
    apply_style()
    t = res.tfr[cond_key]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    vmax = float(np.percentile(np.abs(t["power_pct"]), 99)) or 1.0
    mesh = ax.pcolormesh(
        t["times"] * 1000, t["freqs"], t["power_pct"], cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), shading="nearest",
    )
    ax.axvline(0, color=PALETTE["ink"], lw=1.0)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Time from button TTL (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"{CONDITION_BY_KEY[cond_key].label} — induced power, magnetometers "
        f"({t['n_epochs']} presses)"
    )
    ax.grid(visible=False)
    ax.text(
        0.5, -0.32,
        "A broadband vertical stripe at t = 0 would indicate a switching artifact. "
        "The reference is the mean of the epoch, because the press rate leaves no "
        "clean pre-stimulus window.",
        transform=ax.transAxes, ha="center", va="top",
        color=PALETTE["muted"], fontsize=7.5,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("% change from epoch mean", fontsize=8, color=PALETTE["ink_secondary"])
    cbar.ax.tick_params(labelsize=7, colors=PALETTE["muted"])
    cbar.outline.set_visible(False)
    _despine(ax)
    return to_b64(fig)


def fig_motion_cycle(res) -> str:
    """The movement artifact, averaged over cycles and mapped onto the helmet."""
    apply_style()
    m = res.motion["motion"]
    cond = CONDITION_BY_KEY["motion"]
    info = res.topo_info["mag"]
    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 2.1, 1.25], hspace=0.45, wspace=0.28)
    for row, sensor in enumerate(SENSORS):
        d = m[sensor.key]
        t = d["times"]
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t, d["evoked"].T, color=PALETTE["ink_secondary"], lw=0.4, alpha=0.25)
        worst = d["ch_names"].index(d["worst_channel"])
        ax.plot(t, d["evoked"][worst], color=cond.color, lw=1.8,
                label=f"{d['worst_channel']} (largest)")
        ax.axvline(0, color=PALETTE["ink"], lw=0.9)
        ax.set_ylabel(sensor.unit)
        ax.set_title(f"{sensor.label} — cycle-locked field (<4 Hz), n={d['n_epochs']}")
        ax.legend(loc="upper left", fontsize=8)
        _despine(ax)
        if row == 1:
            ax.set_xlabel("Time from button press at the face (s)")

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(t, d["gfp"], color=cond.color, lw=1.7)
        ax2.fill_between(t, 0, d["gfp"], color=cond.color, alpha=0.18, lw=0)
        ax2.axvline(0, color=PALETTE["ink"], lw=0.9)
        ax2.set_ylabel(f"Field power ({sensor.unit})")
        ax2.set_title("Cycle-locked field power")
        _despine(ax2)
        if row == 1:
            ax2.set_xlabel("Time from button press at the face (s)")

        ax3 = fig.add_subplot(gs[row, 2])
        v = d["p2p_per_channel"]
        if sensor.key == "grad":
            v = collapse_grad_pairs(v, d["ch_names"], res.mag_names)
        im, _ = mne.viz.plot_topomap(
            v, info, axes=ax3, show=False, cmap="Blues", sensors=False,
            contours=4, outlines="head", extrapolate="head",
        )
        ax3.set_title(f"Cycle p-p ({sensor.unit})", fontsize=8.5)
        cbar = fig.colorbar(im, ax=ax3, fraction=0.05, pad=0.04)
        cbar.ax.tick_params(labelsize=6.5, colors=PALETTE["muted"])
        cbar.outline.set_visible(False)
    fig.suptitle(
        "Moved to face — movement artifact locked to the movement cycle",
        fontsize=11, y=0.98, color=PALETTE["ink"],
    )
    return to_b64(fig)
