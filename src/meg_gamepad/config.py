"""Configuration: runs, conditions, bands, palette and analysis parameters.

Everything a reader might want to change lives here, so the analysis modules
stay free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PKG_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PKG_ROOT / "data"
REPORT_DIR = PKG_ROOT / "reports"
CACHE_PATH = REPORT_DIR / "_metrics_cache.pkl"

#: Dataset used when the command line names none. A dataset is a directory
#: under data/ that holds one site's four recordings plus a dataset.json.
DEFAULT_SITE = "tomov_mgh"


# --------------------------------------------------------------------------
# Experimental conditions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One experimental condition.

    This describes the design, not one site's recording of it. File names and
    trigger wiring are site-specific and live in the dataset manifest instead
    (see :mod:`meg_gamepad.dataset`).
    """

    key: str
    label: str
    description: str
    color: str
    gamepad_on: bool
    #: STI101 bit values that mark button presses in this run (empty = none)
    button_codes: tuple[int, ...] = ()
    #: Run this condition is *graded* against. For the two static conditions the
    #: no-controller baseline is the right reference. For the moving condition it
    #: is the passive run: both have the controller present and powered, so the
    #: only difference left is the movement itself. Grading motion against the
    #: baseline instead would confound it with the drift difference between two
    #: recordings twelve minutes apart.
    reference: str = "baseline"


#: Colors are the data-viz reference palette's first three categorical slots
#: (blue / orange / aqua -- the subset documented as all-pairs CVD-validated),
#: with the baseline drawn in neutral secondary ink because it is a reference,
#: not a peer series.
BASELINE_KEY = "baseline"

CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="baseline",
        label="Baseline",
        description="The participant is in the helmet. No controller is present.",
        color="#52514e",
        gamepad_on=False,
    ),
    Condition(
        key="passive",
        label="Held passively",
        description=(
            "The participant holds the powered controller still, with the hands "
            "at the hips."
        ),
        color="#2a78d6",
        gamepad_on=True,
    ),
    Condition(
        key="playing",
        label="Buttons pressed",
        description=(
            "The participant holds the controller still and presses the buttons."
        ),
        color="#eb6834",
        gamepad_on=True,
        button_codes=(512, 2048),
        reference="baseline",
    ),
    Condition(
        key="motion",
        label="Moved to face",
        description=(
            "The participant lifts the controller to the face and returns it to "
            "the hips. This cycle repeats. The participant presses a button at "
            "the face."
        ),
        color="#1baf7a",
        gamepad_on=True,
        button_codes=(2048,),
        reference="passive",
    ),
)

CONDITION_BY_KEY = {c.key: c for c in CONDITIONS}
NON_BASELINE = tuple(c for c in CONDITIONS if c.key != BASELINE_KEY)


# --------------------------------------------------------------------------
# Sensor types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorType:
    key: str
    label: str
    #: multiply raw SI data by this to reach `unit`
    scale: float
    unit: str
    #: typical peak-to-peak rejection threshold in `unit`, for the data-loss metric
    reject_p2p: float


SENSORS: tuple[SensorType, ...] = (
    SensorType("mag", "Magnetometers", 1e15, "fT", 4000.0),
    SensorType("grad", "Gradiometers", 1e13, "fT/cm", 4000.0),
)
SENSOR_BY_KEY = {s.key: s for s in SENSORS}


# --------------------------------------------------------------------------
# Frequency bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    key: str
    label: str
    fmin: float
    fmax: float


#: Physiological bands below the first mains harmonic. These do not depend on
#: the local mains frequency.
_FIXED_BANDS: tuple[Band, ...] = (
    Band("delta", "Delta (1-4 Hz)", 1.0, 4.0),
    Band("theta", "Theta (4-8 Hz)", 4.0, 8.0),
    Band("alpha", "Alpha (8-13 Hz)", 8.0, 13.0),
    Band("beta", "Beta (13-30 Hz)", 13.0, 30.0),
)

#: Guard interval kept clear on each side of a mains harmonic, in Hz.
LINE_GUARD_HZ = 2.0


def bands_for(line_freq: float, fmax: float = 330.0) -> tuple[Band, ...]:
    """Build the band set for a site, given its mains frequency.

    The higher bands step around the mains frequency and its harmonics, so that
    powerline noise never enters a band that the report calls physiological. A
    60 Hz site and a 50 Hz site therefore get different band limits, and both
    get bands that are clean.
    """
    guard = LINE_GUARD_HZ
    gamma_hi = line_freq - guard
    bands = list(_FIXED_BANDS)
    bands.append(
        Band("gamma_low", f"Low gamma (30-{gamma_hi:g} Hz)", 30.0, gamma_hi)
    )
    bands.append(
        Band(
            "gamma_high",
            f"High gamma ({line_freq + guard:g}-100 Hz)",
            line_freq + guard,
            100.0,
        )
    )
    bands.append(Band("hf", "HF (100-200 Hz)", 100.0, 200.0))
    bands.append(Band("vhf", f"VHF (200-{fmax:g} Hz)", 200.0, fmax))
    return tuple(bands)


#: Default band set, for a 60 Hz site. The analysis replaces this with the set
#: that matches the mains frequency recorded in the files.
BANDS: tuple[Band, ...] = bands_for(60.0)


#: Time-domain amplitude windows (filter passbands) for the RMS table.
AMP_WINDOWS: tuple[Band, ...] = (
    Band("drift", "Drift (0.1-1 Hz)", 0.1, 1.0),
    Band("analysis", "Analysis band (1-45 Hz)", 1.0, 45.0),
    Band("high", "High freq (45-330 Hz)", 45.0, 330.0),
)


# --------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------


@dataclass
class Params:
    #: common analysis window, seconds from file start -- all runs are >= 61 s
    crop_tmin: float = 0.5
    crop_duration: float = 60.0

    #: Welch PSD. A 10 s window gives 0.1 Hz resolution, enough to resolve
    #: narrow instrument lines rather than smearing them into the floor.
    psd_n_fft: int = 10_000
    psd_overlap: int = 5_000
    psd_fmin: float = 0.5
    psd_fmax: float = 330.0

    #: spectral-line detection
    line_floor_hz: float = 1.5  # half-width of the local-floor median filter
    line_min_prominence_db: float = 3.0
    line_excess_db: float = 3.0  # prominence gain over baseline to call it new
    powerline_hz: float = 60.0
    powerline_tol_hz: float = 1.0
    #: Below this frequency, narrowband peaks are dominated by physiology (alpha,
    #: mu, beta) and by movement, so a gain there is not evidence of an instrument
    #: line. Lines below it are still listed, but labelled and excluded from the
    #: verdict.
    line_verdict_fmin: float = 30.0

    #: sliding-window amplitude time course
    win_sec: float = 1.0
    win_step_sec: float = 0.25

    #: epoch-rejection ("data loss") metric. Rejection is applied to data
    #: band-passed the way a real pipeline would be, because on unfiltered
    #: magnetometers the criterion just measures environmental drift.
    epoch_sec: float = 1.0
    reject_band: tuple[float, float] = (1.0, 45.0)

    #: button-locked analysis
    ev_tmin: float = -0.3
    ev_tmax: float = 0.5
    ev_baseline: tuple[float, float] = (-0.3, -0.1)
    #: fine-timescale window for detecting a switching transient at the TTL edge
    fine_tmin: float = -0.025
    fine_tmax: float = 0.025
    fine_baseline: tuple[float, float] = (-0.025, -0.008)
    n_surrogate: int = 200
    #: High-pass applied before the drift-free version of the evoked test. Slow
    #: drift that is not locked to anything still survives averaging when events
    #: are regularly paced, so the decisive test is whether a response survives
    #: removal of the drift band.
    hp_evoked_hz: float = 2.0

    #: motion-cycle analysis
    motion_tmin: float = -2.0
    motion_tmax: float = 2.0
    motion_lowpass: float = 4.0

    #: Time-frequency. The wavelets are computed on a window much wider than the
    #: one displayed: a Morlet wavelet needs several of its own standard
    #: deviations of data on each side, and without that padding the convolution
    #: runs off the end of the epoch and produces large edge artifacts. Those
    #: artifacts would otherwise land inside the baseline window and corrupt the
    #: whole map.
    tfr_fmin: float = 4.0
    tfr_fmax: float = 120.0
    tfr_n_freqs: int = 40
    tfr_decim: int = 4
    tfr_pad_tmin: float = -1.2
    tfr_pad_tmax: float = 1.2
    #: Window the percent-change reference is taken over, after the padding (and
    #: its edge artifacts) is trimmed away but before the display crop.
    tfr_ref_tmin: float = -0.9
    tfr_ref_tmax: float = 0.9
    #: n_cycles = f / tfr_cycle_ratio, clipped. The cap stops the wavelet from
    #: growing to half a second at high frequencies, which both smears the time
    #: axis and makes the padding requirement worse.
    tfr_cycle_ratio: float = 2.0
    tfr_cycles_min: float = 3.0
    tfr_cycles_max: float = 15.0

    random_seed: int = 0


PARAMS = Params()


# --------------------------------------------------------------------------
# Report palette (data-viz reference instance, light surface)
# --------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "diverging_pos": "#d03b3b",
    "diverging_neg": "#2a78d6",
    "diverging_mid": "#f0efec",
}


# --------------------------------------------------------------------------
# Verdict thresholds -- the rule table behind the summary cards
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerdictRule:
    key: str
    question: str
    #: dB (or percentage-point) thresholds for "negligible / minor / material"
    minor: float
    material: float
    unit: str = "dB"


VERDICT_RULES: tuple[VerdictRule, ...] = (
    VerdictRule("analysis_band", "Excess power in the 1-45 Hz analysis band", 1.0, 3.0),
    VerdictRule("high_band", "Excess broadband power above 45 Hz", 0.5, 1.5),
    VerdictRule("drift", "Excess sub-1 Hz drift amplitude", 2.0, 6.0),
    VerdictRule(
        "new_lines",
        "Strongest narrowband line gain (>30 Hz, non-powerline)",
        3.0,
        6.0,
    ),
    VerdictRule(
        "epoch_loss",
        "Extra epochs lost to amplitude rejection",
        5.0,
        15.0,
        unit="pp",
    ),
)

SEVERITY_ORDER = ("negligible", "minor", "material")
SEVERITY_COLOR = {
    "negligible": PALETTE["good"],
    "minor": PALETTE["warning"],
    "material": PALETTE["critical"],
}
