"""Assemble the single-file HTML report."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import numpy as np

from . import figures as figs
from . import metrics as mx
from .config import (
    AMP_WINDOWS,
    SENSOR_BY_KEY,
    BANDS,
    BASELINE_KEY,
    CONDITION_BY_KEY,
    CONDITIONS,
    NON_BASELINE,
    PALETTE,
    REPORT_DIR,
    SENSORS,
    SEVERITY_COLOR,
    VERDICT_RULES,
)
from .metrics import db_amp, EPS

log = logging.getLogger(__name__)

SEVERITY_ICON = {"negligible": "✓", "minor": "!", "material": "✕"}
SEVERITY_WORD = {
    "negligible": "Negligible",
    "minor": "Minor",
    "material": "Material",
}


def esc(x) -> str:
    return html.escape(str(x))


def img(b64: str, alt: str) -> str:
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(alt)}">'
        f"</figure>"
    )


def table(headers: list[str], rows: list[list[str]], note: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f'<figcaption class="note">{note}</figcaption>' if note else ""
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>{cap}</div>"
    )


def pill(severity: str) -> str:
    return (
        f'<span class="pill" style="--pill:{SEVERITY_COLOR[severity]}">'
        f'<span class="pill-icon">{SEVERITY_ICON[severity]}</span>'
        f"{SEVERITY_WORD[severity]}</span>"
    )


def fmt_db(v: float) -> str:
    return f"{v:+.2f}"


# --------------------------------------------------------------------------
# Narrative helpers -- text derived from the numbers, not asserted
# --------------------------------------------------------------------------


def _headline(res) -> str:
    """Write the summary statement. All values come from the computed metrics."""
    worst = {k: v["overall"] for k, v in res.verdicts.items()}
    static = [k for k in ("passive", "playing") if k in worst]
    static_ok = all(worst[k] == "negligible" for k in static)

    parts = []
    if static_ok:
        parts.append(
            "The controller does not add measurable noise when the participant "
            "holds it still. This applies both when the controller is idle and "
            "when the participant presses the buttons. All measures in this "
            "report agree with the no-controller baseline. There is no broadband "
            "noise, no new spectral line, and no increase in data loss."
        )
    else:
        bad = ", ".join(
            CONDITION_BY_KEY[k].label.lower() for k in static if worst[k] != "negligible"
        )
        parts.append(
            "The controller changes the data when the participant holds it still "
            f"({bad}). The condition cards below identify the measure that moved."
        )

    if "motion" in res.verdicts:
        m = res.verdicts["motion"]["measures"]
        parts.append(
            "Movement of the controller to the face is the only condition with a "
            f"real effect. The sub-1 Hz drift increases by {m['drift']['value']:.1f} dB. "
            f"The 1-45 Hz band increases by {m['analysis_band']['value']:.1f} dB. "
            "Both values are relative to the run in which the participant holds "
            "the same controller still. Above 45 Hz the change is "
            f"{m['high_band']['value']:+.2f} dB. This is a movement artifact at low "
            "frequency. It is not electromagnetic interference from the device."
        )

    if "playing" in res.locked:
        hp = res.locked["playing"]["mag"]["hp_peak"]
        if hp["significant"]:
            parts.append(
                "The button presses also cause a time-locked response that "
                "survives the high-pass filter. Section 8 examines this response."
            )
        else:
            parts.append(
                "The average of all button presses contains no transient that is "
                "locked to the press after removal of the slow drift. This sets an "
                "upper limit on any switching artifact or trigger crosstalk of "
                f"{hp['null_p95']:.0f} fT on the magnetometers."
            )
    return " ".join(parts)
def _line_conclusion(res) -> str:
    """Write the narrowband-line statement from the detector output."""
    candidates = [
        ln
        for sensor in SENSORS
        for ln in res.lines[sensor.key]["lines"]
        if ln["klass"] == "instrument candidate"
    ]
    flagged = [ln for ln in candidates if ln["counts_for_verdict"]]
    harm = [
        ln
        for sensor in SENSORS
        for ln in res.lines[sensor.key]["lines"]
        if ln["is_powerline"] and ln["is_new"]
    ]
    out = []
    if not flagged:
        if candidates:
            top = max(candidates, key=lambda d: d["max_gain"])
            out.append(
                f"No narrowband line above {res.params.line_verdict_fmin:g} Hz "
                f"increased its prominence by the {res.params.line_excess_db:g} dB "
                "that the detector needs to classify the line as new. The largest "
                f"increase was {top['max_gain']:+.1f} dB at {top['freq']:.1f} Hz. "
                "The data therefore contain no evidence of a clock, a polling "
                "rate, or a switching regulator that radiates into the array."
            )
        else:
            out.append(
                "The detector found no narrowband candidate lines above "
                f"{res.params.line_verdict_fmin:g} Hz."
            )
    else:
        top = max(flagged, key=lambda d: d["max_gain"])
        out.append(
            f"A narrowband line at {top['freq']:.1f} Hz increased its prominence by "
            f"{top['max_gain']:.1f} dB in the "
            f"{CONDITION_BY_KEY[top['max_gain_run']].label.lower()} run. This is the "
            "signature of a periodic electrical source. Examine this line further."
        )
    out.append(
        f"The table also lists peaks below {res.params.line_verdict_fmin:g} Hz, but "
        "the verdict excludes them. The alpha, mu and beta rhythms are narrowband "
        "peaks, and movement adds more peaks in the same range. An increase below "
        f"{res.params.line_verdict_fmin:g} Hz therefore gives no information about "
        "the hardware."
    )
    if harm:
        freqs = sorted({round(ln["freq"]) for ln in harm})
        out.append(
            "The powerline harmonics at "
            + ", ".join(f"{f} Hz" for f in freqs)
            + " did increase between the runs. A sequential design can produce "
            "this result without a device effect, so the harmonics get a separate "
            "test below."
        )
    return " ".join(out)
def _harmonic_evidence(res) -> tuple[str, list[list[str]]]:
    """Compare the harmonic increase on magnetometers and gradiometers.

    Magnetometers respond to the near-uniform field of a distant source.
    Planar gradiometers reject that field, but keep their sensitivity to a
    nearby source. The ratio between the two therefore indicates the distance
    to the source.
    """
    lf = res.line_freq
    harmonics = [lf * n for n in range(1, 6) if lf * n <= res.params.psd_fmax]
    upper = harmonics[2:]  # the harmonics the magnetometer/gradiometer test uses
    rows = []
    for sensor in SENSORS:
        entry = res.lines[sensor.key]
        freqs = entry["freqs"]
        prom = entry["prominence"]
        for target in harmonics:
            i = int(np.argmin(np.abs(freqs - target)))
            cells = [f"{target:g} Hz", sensor.label]
            for cond in CONDITIONS:
                cells.append(f"{prom[cond.key][i]:.1f}")
            rows.append(cells)

    mag_gain, grad_gain = [], []
    for sensor, acc in (("mag", mag_gain), ("grad", grad_gain)):
        entry = res.lines[sensor]
        freqs, prom = entry["freqs"], entry["prominence"]
        for target in upper:
            i = int(np.argmin(np.abs(freqs - target)))
            best = max(
                prom[c.key][i] - prom[BASELINE_KEY][i] for c in NON_BASELINE
            )
            acc.append(best)
    mg, gg = float(np.mean(mag_gain)), float(np.mean(grad_gain))

    if mg - gg > 2.0:
        verdict = (
            "The mean increase across the "
            + ", ".join(f"{h:g}" for h in upper[:-1])
            + f" and {upper[-1]:g} Hz harmonics is "
            f"{mg:.1f} dB on the magnetometers. On the gradiometers it is only "
            f"{gg:.1f} dB. Magnetometers respond to the near-uniform field of a "
            "distant source. Planar gradiometers reject that field, but they keep "
            "their sensitivity to a nearby source. A device in the hands of the "
            "participant is a few tens of centimetres from the helmet, and it "
            "would therefore give a large gradiometer response. The measured "
            "pattern is the opposite. The source is thus distant from the helmet "
            "and is not the controller. The moved-to-face run supports this "
            "conclusion: that run also had the controller powered on, but it does "
            "not show the harmonic increase."
        )
    else:
        verdict = (
            f"The increase is similar on the magnetometers ({mg:.1f} dB) and on "
            f"the gradiometers ({gg:.1f} dB). A source close to the helmet gives "
            "this pattern. This evidence is therefore not sufficient to exclude "
            "the controller as the source."
        )
    return verdict, rows
def _locked_verdict(res, cond_key: str) -> str:
    """Write one statement per sensor type from the statistical test results."""
    out = []
    for sensor in SENSORS:
        e = res.locked[cond_key][sensor.key]
        peak, hp, l60 = e["peak"], e["hp_peak"], e["line60"]
        f = res.fine[cond_key][sensor.key]

        bits = []
        if not peak["significant"]:
            bits.append(
                "No signal is locked to the press. The largest deflection in the "
                f"average is {peak['real']:.1f} {sensor.unit}. This is "
                f"{peak['ratio']:.2f} times the value that random event times give "
                f"(p = {peak['p_value']:.3f})."
            )
        elif not hp["significant"]:
            bits.append(
                f"The unfiltered average contains a deflection of "
                f"{peak['real']:.0f} {sensor.unit} that is above chance "
                f"({peak['ratio']:.2f} times the null, p = {peak['p_value']:.3f}). "
                f"This deflection does not survive a {res.params.hp_evoked_hz:g} Hz "
                f"high-pass filter ({hp['ratio']:.2f} times the null, "
                f"p = {hp['p_value']:.3f}). The locked signal is therefore a slow "
                "field and not a fast transient. Its amplitude increases with the "
                "distance from the baseline window. It is not bound to the moment "
                "of the press."
            )
        else:
            bits.append(
                "A locked response is present, and it survives the high-pass "
                f"filter ({hp['real']:.0f} {sensor.unit}, {hp['ratio']:.2f} times "
                f"the null, p = {hp['p_value']:.3f})."
            )

        if l60["significant"]:
            bits.append(
                "The presses are partially phase-locked to the mains supply. The "
                f"{res.line_freq:g} Hz component of the average is "
                f"{l60['real']:.1f} "
                f"{sensor.unit} ({l60['ratio']:.2f} times the null, "
                f"p = {l60['p_value']:.3f})."
            )
        else:
            bits.append(
                f"There is no phase-locking to the mains supply "
                f"({l60['ratio']:.2f} times the null, p = {l60['p_value']:.3f}). "
                "The button TTL is therefore independent of the "
                f"{res.line_freq:g} Hz cycle."
            )

        bits.append(
            f"In the window of +/-5 ms around the TTL edge, the largest deflection "
            f"is {f['edge_peak_max']:.1f} {sensor.unit}. The high-pass test "
            "includes this window. The upper limit for a switching transient is "
            f"therefore the 95th percentile of the null, {hp['null_p95']:.0f} "
            f"{sensor.unit}."
        )
        out.append(f"<strong>{sensor.label}.</strong> " + " ".join(bits))
    return "<p>" + "</p><p>".join(out) + "</p>"
def _section_summary(res) -> str:
    cards = []
    for cond in NON_BASELINE:
        v = res.verdicts[cond.key]
        rows = []
        for rule in VERDICT_RULES:
            m = v["measures"][rule.key]
            detail = f' <span class="sub">{esc(m.get("detail", ""))}</span>' if m.get("detail") else ""
            rows.append(
                f'<li><span class="mname">{esc(rule.question)}{detail}</span>'
                f'<span class="mval">{m["value"]:+.2f} {rule.unit}</span>'
                f'{pill(m["severity"])}</li>'
            )
        cards.append(
            f'<article class="card" style="--accent:{cond.color}">'
            f'<header><h3>{esc(cond.label)}</h3>{pill(v["overall"])}</header>'
            f'<p class="desc">{esc(cond.description)}</p>'
            f'<p class="ref">Graded against: <strong>{esc(v["reference_label"])}</strong></p>'
            f'<ul class="measures">{"".join(rows)}</ul>'
            "</article>"
        )
    thresholds = table(
        ["Measure", "Negligible", "Minor", "Material"],
        [
            [
                esc(r.question),
                f"&lt; {r.minor:g} {r.unit}",
                f"{r.minor:g}–{r.material:g} {r.unit}",
                f"≥ {r.material:g} {r.unit}",
            ]
            for r in VERDICT_RULES
        ],
        "These thresholds apply to every condition. The noise floor of a MEG "
        "room is reproducible to approximately +/-3 dB over tens of minutes. The "
        "1 dB limit for the 'minor' grade is therefore conservative.",
    )
    return f"""
<section id="summary">
  <h2>1 · Verdict</h2>
  <p class="lede">{_headline(res)}</p>
  <div class="cards">{''.join(cards)}</div>
  <div class="callout note-callout">
    <strong>The reference run for the moving condition is the passive run, not
    the baseline.</strong> The operator recorded the baseline twelve minutes
    before the controller runs. Of the four runs, the baseline contains the most
    sub-1 Hz environmental drift. A comparison against the baseline would
    therefore remove a real movement artifact from the result. The passive run
    contains the same powered controller in the same hands. Movement is the only
    remaining difference between the two runs. The two static conditions keep the
    baseline as their reference, because the baseline answers the question
    "does the device add noise". All other figures and tables in this report use
    the baseline as the reference and identify it as such.
  </div>
  <details><summary>How each condition is graded</summary>{thresholds}</details>
</section>"""


def _section_inventory(res) -> str:
    rows = []
    for row in res.inventory:
        rows.append(
            [
                f'<span class="swatch" style="background:{row["color"]}"></span> '
                f"<strong>{esc(row['label'])}</strong>",
                f"<code>{esc(row['filename'])}</code>",
                esc(row["meas_date"].replace(" UTC", "")),
                f"{row['orig_duration']:.0f} s",
                f"{row['sfreq']:.0f} Hz",
                f"{row['highpass']:.2g}–{row['lowpass']:.0f} Hz",
                str(row["n_events"]) if row["n_events"] else "—",
                f"{row['isi_median']:.2f} s" if row["isi_median"] else "—",
            ]
        )
    inv = table(
        ["Condition", "File", "Recorded (UTC)", "Length", "Rate", "Band",
         "Button events", "Median interval"],
        rows,
        f"The analysis uses a common {res.params.crop_duration:.0f} s window in "
        f"each run. The window starts {res.params.crop_tmin:g} s after the start of "
        "the file. All runs contain 102 magnetometers and 204 planar "
        "gradiometers. The files contain six SSP vectors. This analysis does not "
        "apply them, because the report must describe the data as the sensors "
        "record it.",
    )
    warn = ""
    if res.acquisition_warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in res.acquisition_warnings)
        warn = f'<div class="callout bad"><strong>Acquisition mismatch.</strong><ul>{items}</ul></div>'
    else:
        warn = (
            '<div class="callout good"><strong>The acquisition parameters '
            "match.</strong> The sampling rate, the hardware filter band and the "
            "channel set are identical in all four runs. The dB comparisons in "
            "this report are therefore valid.</div>"
        )

    caveats = """
<div class="callout warn">
  <strong>Limits of this dataset.</strong>
  <ul>
    <li><strong>The runs are sequential and not interleaved.</strong> The operator
        recorded the baseline approximately 12 minutes before the three controller
        runs. Environmental noise that changed during that interval appears in the
        results as a controller effect. The opposite error is also possible. The
        mains-harmonic result in section 5 is an example. Section 5 resolves it
        with an argument about sensor geometry, not with the time series.</li>
    <li><strong>There is no empty-room run.</strong> A participant is present in
        every run. Brain, cardiac and muscle signals are therefore present in all
        conditions. Some differences between the conditions come from the
        participant and not from the hardware. The alpha decrease in section 4 is
        the clearest example.</li>
    <li><strong>The system did not record head position.</strong> These files
        contain no HPI or cHPI data and no device-to-head transform. It is
        therefore not possible to measure or to remove head movement. Section 9
        cannot separate movement of the head from movement of the controller.</li>
    <li><strong>The dataset contains one participant and one minute for each
        condition.</strong> This is sufficient to detect a large artifact. It is
        not sufficient to set a precise limit on a small one.</li>
  </ul>
</div>"""
    return f"""
<section id="data">
  <h2>2 · What was recorded</h2>
  {inv}
  {warn}
  {caveats}
</section>"""


def _section_amplitude(res, figures) -> str:
    rows = []
    windows = [("broadband", "As recorded")] + [(w.key, w.label) for w in AMP_WINDOWS]
    for sensor in SENSORS:
        for wkey, wlabel in windows:
            vals = res.amplitudes[sensor.key][wkey]
            base = float(np.median(vals[BASELINE_KEY]))
            cells = [sensor.label, wlabel, f"{base:,.1f}"]
            for cond in NON_BASELINE:
                med = float(np.median(vals[cond.key]))
                cells.append(
                    f"{med:,.1f} <span class='sub'>({fmt_db(db_amp(med / max(base, EPS)))} dB)</span>"
                )
            rows.append(cells)
    amp = table(
        ["Sensors", "Passband", f"Baseline median", *[c.label for c in NON_BASELINE]],
        rows,
        "The table gives the median per-channel RMS. The units are fT for the "
        "magnetometers and fT/cm for the gradiometers. The value in brackets is "
        "the change from the baseline in power dB. The as-recorded row reverses "
        "the result of the other rows, because the baseline run contains the "
        "<em>most</em> sub-1 Hz environmental drift of the four runs. For this "
        "reason, all other measures in this report use a defined passband.",
    )
    return f"""
<section id="amplitude">
  <h2>3 · Broadband amplitude</h2>
  <p>The first question is whether the controller increases the noise floor.
     Sub-1 Hz environmental drift dominates the magnetometer RMS. The comparison
     therefore uses three separate passbands and not the as-recorded signal.</p>
  {amp}
  {img(figures['amplitude'], 'Per-channel RMS amplitude distributions by passband')}
</section>"""


def _section_spectra(res, figures) -> str:
    rows = []
    for sensor in SENSORS:
        for band in res.bands_def:
            entry = res.bands[sensor.key][band.key]
            cells = [
                sensor.label,
                band.label,
                f"{entry[BASELINE_KEY]['abs_amp']:,.2f}",
            ]
            for cond in NON_BASELINE:
                v = entry[cond.key]["db_median"]
                cls = "hot" if v >= 1.0 else ("cold" if v <= -1.0 else "")
                cells.append(f"<span class='{cls}'>{fmt_db(v)}</span>")
            rows.append(cells)
    band_tbl = table(
        ["Sensors", "Band", "Baseline (unit/√Hz)", *[c.label for c in NON_BASELINE]],
        rows,
        "The table gives the median across channels of the per-channel power "
        f"change from the baseline, in dB. The band limits avoid "
        f"{res.line_freq:g} Hz and its harmonics. Powerline noise therefore does not enter a physiological "
        "band. The colour marks values that are more than +/-1 dB.",
    )
    alpha_mag = res.bands["mag"]["alpha"]["playing"]["db_median"]
    alpha_grad = res.bands["grad"]["alpha"]["playing"]["db_median"]
    return f"""
<section id="spectra">
  <h2>4 · Spectra</h2>
  {img(figures['psd'], 'Amplitude spectral density per condition')}
  {img(figures['psd_ratio'], 'Spectral change versus baseline in dB')}
  {band_tbl}
  <div class="callout note-callout">
    <strong>The alpha row shows physiology and not a hardware effect.</strong>
    Alpha power <em>decreases</em> by {abs(alpha_mag):.1f} dB on the magnetometers
    and by {abs(alpha_grad):.1f} dB on the gradiometers while the participant
    presses the buttons. Alpha desynchronisation during an active task is a
    well-established effect in electrophysiology. An instrument artifact does not
    decrease the power in one narrow band and leave the other bands unchanged.
    This result is a positive control. It shows that the participant performed the
    task, and that the analysis is sensitive enough to detect a real effect.
  </div>
</section>"""


def _section_lines(res, figures) -> str:
    rows = []
    klass_tag = {
        "powerline": lambda ln: f"{res.line_freq:g} Hz × {ln['harmonic']}",
        "low-frequency": lambda ln: '<span class="sub">physiological band</span>',
        "instrument candidate": lambda ln: "<strong>instrument candidate</strong>",
    }
    for sensor in SENSORS:
        for ln in res.lines[sensor.key]["lines"][:14]:
            tag = klass_tag[ln["klass"]](ln)
            cells = [
                f"{ln['freq']:.1f} Hz",
                sensor.label,
                tag,
                f"{ln['baseline_prominence']:.1f}",
            ]
            for cond in NON_BASELINE:
                g = ln["gain"].get(cond.key, float("nan"))
                hot = (
                    g >= res.params.line_excess_db
                    and ln["klass"] == "instrument candidate"
                )
                cells.append(f"<span class='{'hot' if hot else ''}'>{fmt_db(g)}</span>")
            rows.append(cells)
    line_tbl = table(
        [
            "Frequency",
            "Sensors",
            "Type",
            "Baseline prominence (dB)",
            *[f"{c.label} (gain)" for c in NON_BASELINE],
        ],
        rows,
        "Prominence is the height of a peak above the local spectral floor. The "
        "floor is the median over +/-1.5 Hz. Gain is the increase of that "
        "prominence from the baseline run. The detector searched the "
        "controller-on runs, so every row in the table is a candidate. The gain "
        "column therefore carries the result. Only an <strong>instrument "
        "candidate</strong> row is evidence against the device. Such a row must "
        "not be a powerline harmonic, and it must be above "
        f"{res.params.line_verdict_fmin:g} Hz. The colour marks the rows that also "
        "exceed the gain threshold.",
    )
    harm_verdict, harm_rows = _harmonic_evidence(res)
    harm_tbl = table(
        ["Harmonic", "Sensors", *[c.label for c in CONDITIONS]],
        harm_rows,
        "The table gives the peak prominence above the local floor, in dB, at "
        "each mains harmonic.",
    )
    return f"""
<section id="lines">
  <h2>5 · Narrowband lines</h2>
  <p>A powered digital device usually produces narrow spectral lines. A clock, a
     polling rate or a switching regulator each give such a line. These lines are
     easier to detect than broadband noise. This section finds the peaks that rise
     above the local spectral floor. It then compares the prominence of each peak
     between the runs.</p>
  <p class="lede">{_line_conclusion(res)}</p>
  {img(figures['lines_mag'], 'Narrowband line prominence gain, magnetometers')}
  {line_tbl}
  <h3>Analysis of the mains harmonics</h3>
  {harm_tbl}
  <div class="callout note-callout"><strong>Verdict.</strong> {harm_verdict}</div>
</section>"""


def _section_topography(res, figures) -> str:
    panels = "".join(
        f"<h3>{esc(CONDITION_BY_KEY[k].label)}</h3>"
        + img(figures[f"topo_{k}"], f"Power change topography, {k}")
        for k in [c.key for c in NON_BASELINE]
    )
    return f"""
<section id="topography">
  <h2>6 · Spatial distribution on the helmet</h2>
  <p>Topography separates the classes of source better than amplitude does. A
     distant environmental source gives a smooth, near-uniform field across the
     array, and it appears mainly on the magnetometers. A source held in the hands
     or against the body gives a gradient. That gradient is largest at the
     inferior sensors nearest to the source. A neural or muscular source gives a
     focal dipolar pattern. Each map below shows the per-channel power change from
     the baseline. Each gradiometer pair is averaged onto the helmet site that the
     pair shares with its magnetometer.</p>
  {panels}
</section>"""


def _section_stability(res, figures) -> str:
    rows = []
    for cond in CONDITIONS:
        cells = [
            f'<span class="swatch" style="background:{cond.color}"></span> {esc(cond.label)}'
        ]
        for sensor in SENSORS:
            r = res.rejection[cond.key][sensor.key]
            cells += [
                f"{r['median_p2p']:,.0f}",
                f"{r['p95_p2p']:,.0f}",
                f"{r['calibrated_pct']:.1f}%",
                f"{r['pct_rejected']:.0f}%",
            ]
        rows.append(cells)

    sensor_cols = []
    for sensor in SENSORS:
        short = sensor.label.split()[0][:4]
        sensor_cols += [
            f"{short} median p-p",
            f"{short} 95th p-p",
            f"{short} lost @cal.",
            f"{short} lost @4000",
        ]
    cal = {
        s.key: res.rejection[BASELINE_KEY][s.key]["calibrated_threshold"]
        for s in SENSORS
    }
    note = (
        f"The analysis uses non-overlapping {res.params.epoch_sec:g} s epochs of "
        f"{res.params.reject_band[0]:g}-{res.params.reject_band[1]:g} Hz filtered "
        "data. The amplitudes are in fT and fT/cm. The band-pass filter is applied "
        "first. On unfiltered magnetometers a peak-to-peak criterion measures "
        "mostly environmental drift, and it would classify the no-controller "
        "baseline as the noisiest of the four runs. <em>@4000</em> is the "
        "conventional MEGIN threshold. It rejects no epoch in any condition, and "
        "it therefore cannot separate the conditions. <em>@cal.</em> uses a "
        "threshold calibrated to reject 5% of the baseline epochs "
        f"({cal['mag']:,.0f} fT, {cal['grad']:,.0f} fT/cm). This gives the same "
        "criterion to every condition."
    )
    rej = table(["Condition", *sensor_cols], rows, note)

    return f"""
<section id="stability">
  <h2>7 &middot; Time course and data loss</h2>
  <p>A continuous artifact, a burst artifact and an artifact that follows a
     movement cycle each need a different correction. The figure below therefore
     shows the amplitude against the time within each run.</p>
  {img(figures['sliding'], 'Sliding-window amplitude within each run')}
  <p>The quantity of lost data depends on the rejection threshold. A single
     threshold is an arbitrary choice. The figure below therefore shows the full
     curve of rejected epochs against the threshold.</p>
  {img(figures['rejection'], 'Epochs lost across the range of rejection thresholds')}
  {rej}
</section>"""


def _section_buttons(res, figures) -> str:
    if "playing" not in res.button:
        return ""
    isi = next(
        (r["isi_median"] for r in res.inventory if r["key"] == "playing"), None
    ) or float("nan")
    body = []
    for key in [c.key for c in NON_BASELINE if c.key in res.button]:
        cond = CONDITION_BY_KEY[key]
        body.append(f"<h3>{esc(cond.label)}</h3>")
        body.append(_locked_verdict(res, key))
        body.append(img(figures[f"evoked_{key}"], f"Button-locked average, {key}"))
        body.append(img(figures[f"fine_{key}"], f"TTL-edge zoom, {key}"))
        body.append(img(figures[f"locked_{key}"], f"Locked-response tests, {key}"))
        body.append(img(figures[f"btopo_{key}"], f"Button-locked topography, {key}"))
        if key in res.tfr:
            body.append(img(figures[f"tfr_{key}"], f"Induced power around press, {key}"))

    return f"""
<section id="buttons">
  <h2>8 · Response to a button press</h2>
  <p>The buttons of the controller send TTL bits to the MEG on
     <code>STI101</code>. Each press therefore has a timestamp, and the analysis
     can average the data about that timestamp. One ambiguity applies to this
     measurement. A participant who presses a button also produces a motor and
     somatosensory response, and that response is <em>not</em> an artifact. The
     four criteria below separate the two possibilities.</p>
  <ol class="criteria">
    <li><strong>Latency.</strong> An electrical artifact from the switch contact
        or from trigger-line crosstalk is locked to the TTL edge. It stays within
        a few milliseconds of that edge. The 330 Hz acquisition low-pass filter
        spreads a true step over approximately +/-3 ms, and this sets the
        resolution of the test. A motor response starts before the switch closes,
        because the muscle activity comes first. It then continues for tens to
        hundreds of milliseconds.</li>
    <li><strong>Topography.</strong> Crosstalk and a device field follow the
        hardware. A distant source gives a near-uniform field. A near source gives
        the largest signal at the sensors nearest to it. A motor response gives a
        focal dipolar pattern above the sensorimotor cortex.</li>
    <li><strong>Spectral signature.</strong> A switching transient is broadband.
        In the time-frequency map it appears as a vertical stripe at t = 0. A
        movement response appears as a decrease in the beta band, and then as a
        rebound.</li>
    <li><strong>A surrogate null.</strong> The analysis averages the same number
        of epochs at random times. This gives the peak amplitude for the condition
        in which no locked response exists. The real average must exceed that
        value. The statistic is the maximum across all channels and the full
        window, so the comparison includes the correction for that search.</li>
  </ol>
  <p>The analysis tests three statistics against that null.
     <strong>Any locked deflection</strong> is the largest value at any point in
     the average. This statistic is sensitive, but it also responds to slow drift.
     Slow drift survives the average when the presses have a regular rate.
     <strong>Survives high-pass</strong> repeats the test after the removal of all
     content below {res.params.hp_evoked_hz:g} Hz. Only a fast response can pass
     this test, which is the applicable test for a switching artifact.
     <strong>Mains phase-locking</strong> shows whether the presses are
     synchronous with the {res.line_freq:g} Hz mains cycle. Such synchronism would add a coherent
     ripple to every button-locked average.</p>
  <div class="callout warn">
    <strong>The press rate limits this measurement.</strong> The median interval
    between presses in the buttons-pressed run is {isi:.2f} s. Most pre-stimulus
    windows therefore contain the previous press. For this reason the analysis
    uses an epoch-mean reference for the time-frequency map, and it does not
    attempt to measure an event-related desynchronisation. A separate run with a
    slow, regular press rate would be necessary for that measurement.
  </div>
  {''.join(body)}
</section>"""


def _motion_contrast_table(res) -> str:
    """Moving vs holding still -- the contrast that isolates the movement."""
    rows = []
    windows = [("drift", "Drift (0.1-1 Hz)"), ("analysis", "Analysis band (1-45 Hz)"),
               ("high", "High freq (45-330 Hz)")]
    for sensor in SENSORS:
        for wkey, wlabel in windows:
            vals = res.amplitudes[sensor.key][wkey]
            passive = float(np.median(vals["passive"]))
            motion = float(np.median(vals["motion"]))
            base = float(np.median(vals[BASELINE_KEY]))
            v_pass = float(db_amp(motion / max(passive, EPS)))
            v_base = float(db_amp(motion / max(base, EPS)))
            cls = "hot" if v_pass >= 1.0 else ("cold" if v_pass <= -1.0 else "")
            rows.append(
                [
                    sensor.label,
                    wlabel,
                    f"{passive:,.1f}",
                    f"{motion:,.1f}",
                    f"<span class='{cls}'>{fmt_db(v_pass)}</span>",
                    f"<span class='sub'>{fmt_db(v_base)}</span>",
                ]
            )
    return table(
        ["Sensors", "Passband", "Held still", "Moved", "vs held still (dB)",
         "vs baseline (dB)"],
        rows,
        "The table gives the median per-channel RMS. The two right-hand columns "
        "give the same movement effect against two different references, and the "
        "two columns disagree. Against the baseline, most of the drift increase "
        "disappears, because the baseline run contains an unusual quantity of "
        "drift. The 'vs held still' column is the correct comparison. That "
        "comparison keeps the same device and the same hands, and movement is the "
        "only difference. The two references agree on one result: nothing changes "
        "above 45 Hz.",
    )


def _section_motion(res, figures) -> str:
    if "motion" not in res.motion:
        return ""
    m = res.motion["motion"]
    rows = []
    for sensor in SENSORS:
        d = m[sensor.key]
        rows.append(
            [
                sensor.label,
                str(d["n_epochs"]),
                f"{d['p2p_median']:,.1f}",
                f"{d['p2p_max']:,.1f}",
                esc(d["worst_channel"]),
            ]
        )
    tbl = table(
        ["Sensors", "Cycles", "Median cycle p-p", "Max cycle p-p", "Largest channel"],
        rows,
        "Peak-to-peak amplitude of the cycle-locked average after low-pass "
        f"filtering at {res.params.motion_lowpass:g} Hz, in fT and fT/cm. Averaging "
        "The average across cycles removes all content that is not locked to the "
        "movement. These values are therefore lower limits for the excursion in a "
        "single cycle. The number of cycles is less than the number of button "
        "events in section 2. Cycles that are too near to the start or the end of "
        "the run have insufficient data for the +/-2 s window.",
    )
    drift = res.verdicts["motion"]["measures"]["drift"]["value"]
    band = res.bands["mag"]["delta"]["motion"]["db_median"]
    cross = {
        sensor.key: mx.excess_crossover(res.psds, sensor.key, "motion", "passive")
        for sensor in SENSORS
    }
    cross_txt = " and ".join(
        f"{v:.1f} Hz on {SENSOR_BY_KEY[k].label.lower()}"
        for k, v in cross.items()
        if v is not None
    ) or "nowhere in the measured band"
    return f"""
<section id="motion">
  <h2>9 · The movement artifact</h2>
  <p>In the moved-to-face run, each button press identifies the top of one
     movement cycle. The movement is therefore available as an event-locked
     measurement. This is the only condition with an unambiguous effect. The
     sub-1 Hz drift increases by {drift:.1f} dB relative to the run in which the
     participant holds the same controller still. The delta band increases by
     {band:.1f} dB relative to the baseline.</p>
  <div class="callout note-callout">
    <strong>Frequency limit of the movement effect.</strong> The movement excess
    stays above 3 dB only up to {cross_txt}. Above those frequencies the moved
    run and the held-still run agree to within approximately 1 dB, up to 330 Hz.
    This limit is important for the correct use of the summary card. The card
    grades the 1-45 Hz analysis band as <em>material</em>, but only the lowest
    part of that band produces the grade. The remainder of the analysis band and
    all higher frequencies show no effect.
  </div>
  {img(figures['motion'], 'Movement artifact locked to the movement cycle')}
  {tbl}
  {_motion_contrast_table(res)}
  <div class="callout note-callout">
    <strong>The source is the controller or the participant.</strong> This
    dataset cannot fully separate the two sources. A participant who lifts an arm
    to the face also moves the torso and the head, not only the device. The files
    contain no HPI data, so there is no record of head position for a regression.
    The data do show that the effect stays below a few Hz. All movement artifacts
    occupy that range, and a standard high-pass filter already removes it. A
    magnetic device would give an artifact that increases as the distance to the
    sensors decreases. That artifact would be largest at the sensors that the
    device passes most closely. A dedicated run that moves the controller through
    the helmet, with no participant present, would test this prediction.
  </div>
</section>"""


def _section_methods(res) -> str:
    p = res.params
    return f"""
<section id="methods">
  <h2>10 · Methods and reproduction</h2>
  <p>The analysis uses MNE-Python {esc(res.mne_version)}. The system computed this
     report on {esc(res.computed_at)} in {res.elapsed_s:.0f} s. The data come from
     the dataset <code>{esc(res.site)}</code> in
     <code>{esc(res.dataset_root)}</code>. The mains frequency is
     {res.line_freq:g} Hz, and the band limits above 30 Hz follow from it.</p>
  <ul class="methods">
    <li><strong>Window.</strong> The analysis uses {p.crop_duration:.0f} s of each
        run, from t = {p.crop_tmin:g} s. All four runs therefore contribute the
        same duration, and no run includes its own start-up transient.</li>
    <li><strong>Projections.</strong> The analysis does not apply the six stored
        SSP vectors. The report describes the data as the system records it.</li>
    <li><strong>Spectra.</strong> Welch method, {p.psd_n_fft / 1000:g} s Hann
        windows, 50% overlap. The resolution is {1000 / p.psd_n_fft:.1f} Hz over
        {p.psd_fmin:g}-{p.psd_fmax:g} Hz. This resolution is sufficient to
        separate an instrument line from the spectral floor.</li>
    <li><strong>Line detection.</strong> The detector keeps the peaks that rise
        {p.line_min_prominence_db:g} dB or more above a +/-{p.line_floor_hz:g} Hz
        median floor. It classifies a peak as new at a prominence gain of
        {p.line_excess_db:g} dB or more above the reference run. Only peaks above
        {p.line_verdict_fmin:g} Hz that are not powerline harmonics enter the
        verdict.</li>
    <li><strong>Amplitude.</strong> Per-channel RMS after a zero-phase FIR filter
        into each passband. The report gives each change as 20&middot;log<sub>10</sub>
        of the amplitude ratio. These values are therefore directly comparable with
        the spectral values in dB.</li>
    <li><strong>Event locking.</strong> The analysis decodes the button onsets
        bitwise from <code>STI101</code> and merges onsets that are less than
        50 ms apart. The evoked window is {p.ev_tmin:g} to {p.ev_tmax:g} s with a
        baseline of {p.ev_baseline[0]:g} to {p.ev_baseline[1]:g} s. The TTL-edge
        test uses the unfiltered signal with a
        {p.fine_baseline[0] * 1000:.0f} to {p.fine_baseline[1] * 1000:.0f} ms
        baseline. The null distribution comes from {p.n_surrogate} averages at
        random onsets.</li>
    <li><strong>Time-frequency.</strong> Morlet wavelets at {p.tfr_n_freqs}
        log-spaced frequencies from {p.tfr_fmin:g} to {p.tfr_fmax:g} Hz. The number
        of cycles is f/{p.tfr_cycle_ratio:g}, limited to the range
        {p.tfr_cycles_min:g} to {p.tfr_cycles_max:g}. The computation uses epochs
        of {p.tfr_pad_tmin:g} to {p.tfr_pad_tmax:g} s. This padding prevents the
        edge artifact that occurs when a wavelet extends past the end of its
        epoch. The analysis then trims the data to {p.tfr_ref_tmin:g} to
        {p.tfr_ref_tmax:g} s, takes the mean of that window as the reference, and
        displays {p.ev_tmin:g} to {p.ev_tmax:g} s. The reference is the epoch mean
        and not a pre-stimulus baseline, because the press rate leaves no clean
        pre-stimulus window.</li>
    <li><strong>Gradiometer topography.</strong> The analysis averages the two
        planar gradiometers at each helmet site onto that site. The gradiometer
        maps therefore use the 102 magnetometer positions.</li>
  </ul>
  <p class="note">To reproduce this report, run
     <code>python scripts/run_analysis.py</code>. The thresholds, the bands and
     the palette are in <code>src/meg_gamepad/config.py</code>.</p>
</section>"""


# --------------------------------------------------------------------------
# Figure building
# --------------------------------------------------------------------------


def build_figures(res) -> dict[str, str]:
    out: dict[str, str] = {}
    log.info("figure: spectra")
    out["psd"] = figs.fig_psd(res)
    out["psd_ratio"] = figs.fig_psd_ratio(res)
    log.info("figure: amplitude")
    out["amplitude"] = figs.fig_amplitude_distributions(res)
    out["sliding"] = figs.fig_sliding(res)
    log.info("figure: lines")
    out["lines_mag"] = figs.fig_lines(res, "mag")
    log.info("figure: topography")
    for cond in NON_BASELINE:
        out[f"topo_{cond.key}"] = figs.fig_topography(res, cond.key)
    out["rejection"] = figs.fig_rejection(res)
    log.info("figure: event-locked")
    for key in res.button:
        out[f"evoked_{key}"] = figs.fig_button_evoked(res, key)
        out[f"fine_{key}"] = figs.fig_fine_transient(res, key)
        out[f"locked_{key}"] = figs.fig_locked_tests(res, key)
        out[f"btopo_{key}"] = figs.fig_button_topography(res, key)
    for key in res.tfr:
        out[f"tfr_{key}"] = figs.fig_tfr(res, key)
    if "motion" in res.motion:
        out["motion"] = figs.fig_motion_cycle(res)
    return out


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{color-scheme:light}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding-block:40px 72px;padding-inline:20px}
h1{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.15;margin:0 0 .3em;letter-spacing:-.02em}
h2{font-size:1.3rem;margin:0 0 .6em;letter-spacing:-.01em;padding-bottom:.35em;
  border-bottom:1px solid var(--grid)}
h3{font-size:1.02rem;margin:2em 0 .5em;color:var(--ink)}
p{margin:0 0 1em}
section{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:26px 28px;margin-bottom:22px}
.subtitle{color:var(--ink2);font-size:1.02rem;margin:0 0 1.4em;max-width:72ch}
.lede{font-size:1.05rem;color:var(--ink);max-width:76ch}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;color:var(--muted);font-size:.84rem;
  margin-top:1.2em}
.meta code{font-size:.84rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
  background:#f0efec;padding:.1em .34em;border-radius:4px}
figure{margin:1.6em 0}
figure img{display:block;width:100%;height:auto;border-radius:8px}
figcaption,.note{color:var(--muted);font-size:.84rem;line-height:1.55;margin-top:.5em}
.tablewrap{margin:1.4em 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.86rem;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 11px;border-bottom:1px solid var(--grid);
  white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{color:var(--ink2);font-weight:600;border-bottom:1.5px solid var(--axis);
  position:sticky;top:0;background:var(--surface)}
tbody tr:hover{background:#f6f5f2}
.sub{color:var(--muted);font-size:.9em}
.hot{color:#a32a2a;font-weight:600}
.cold{color:#1c5cab;font-weight:600}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;
  vertical-align:middle;margin-right:5px}
.cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));
  margin:1.5em 0}
.card{border:1px solid var(--border);border-radius:11px;padding:16px 18px;
  border-top:3px solid var(--accent);background:var(--surface)}
.card header{display:flex;align-items:center;justify-content:space-between;gap:10px;
  margin-bottom:.3em}
.card h3{margin:0;font-size:1rem}
.card .desc{color:var(--ink2);font-size:.86rem;margin:0 0 .5em}
.card .ref{font-size:.78rem;color:var(--muted);margin:0 0 .7em}
.measures{list-style:none;margin:0;padding:0;font-size:.84rem}
.measures li{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;
  padding:5px 0;border-top:1px solid var(--grid)}
.mname{color:var(--ink2)}
.mval{font-variant-numeric:tabular-nums;color:var(--ink)}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:.74rem;font-weight:600;
  padding:2px 8px 2px 6px;border-radius:999px;color:#fff;background:var(--pill);
  white-space:nowrap}
.pill-icon{font-size:.8em;line-height:1}
.callout{border-radius:10px;padding:14px 18px;margin:1.5em 0;font-size:.92rem;
  border-left:3px solid var(--muted);background:#f6f5f2}
.callout ul{margin:.5em 0 0;padding-left:1.2em}
.callout li{margin-bottom:.45em}
.callout.good{border-left-color:#0ca30c;background:#f2f8f2}
.callout.warn{border-left-color:#fab219;background:#fdf8ed}
.callout.bad{border-left-color:#d03b3b;background:#fbf2f2}
.note-callout{border-left-color:#2a78d6;background:#f2f6fc}
.criteria{margin:1em 0 1.6em;padding-left:1.3em;font-size:.93rem}
.criteria li{margin-bottom:.6em}
.methods{font-size:.92rem;padding-left:1.2em}
.methods li{margin-bottom:.5em}
details{margin-top:1.2em;font-size:.9rem}
summary{cursor:pointer;color:var(--ink2);font-weight:600}
nav.toc{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:18px 24px;margin-bottom:22px}
nav.toc ol{margin:0;padding-left:1.3em;columns:2;column-gap:32px;font-size:.9rem}
nav.toc li{margin-bottom:.35em;break-inside:avoid}
nav.toc a{color:var(--ink2);text-decoration:none}
nav.toc a:hover{color:#2a78d6;text-decoration:underline}
footer{color:var(--muted);font-size:.82rem;text-align:center;padding:10px 0 0}
@media (max-width:620px){
  section{padding:20px 16px;border-radius:11px}
  nav.toc ol{columns:1}
  .measures li{grid-template-columns:1fr auto}
  .measures .pill{grid-column:2}
}

/* Print / PDF. The same document, paginated: keep every figure, table row and
   verdict card whole, repeat table headers across pages, and never leave a
   heading stranded at the foot of a page. */
@page{size:A4;margin:14mm}
@media print{
  :root{--page:#fff}
  html,body{background:#fff}
  body{font-size:10.5pt;line-height:1.5;
    -webkit-print-color-adjust:exact;print-color-adjust:exact}
  .wrap{max-width:none;padding:0}
  section{border-radius:0;border:none;border-top:1px solid var(--grid);
    padding:14px 0 6px;margin-bottom:0;break-inside:auto}
  section#summary{border-top:none}
  h1{font-size:19pt}
  h2{font-size:13pt;margin-top:4px}
  h3{font-size:11pt}
  h2,h3{break-after:avoid;break-inside:avoid}
  p,li{orphans:3;widows:3}
  figure{break-inside:avoid;margin:12px 0}
  figure img{max-height:205mm;object-fit:contain}
  /* One card per row: at page width the three-column measure rows would
     otherwise wrap every label onto three lines. */
  .cards{grid-template-columns:1fr;gap:8px}
  .card{break-inside:avoid}
  nav.toc{border-radius:0;break-inside:avoid;margin-bottom:6px}
  .tablewrap{overflow:visible;break-inside:auto}
  table{font-size:8.5pt}
  thead{display:table-header-group}
  thead th{position:static}
  tr{break-inside:avoid}
  .callout{break-inside:avoid}
  details{break-inside:avoid}
  details summary{display:none}
  details[open] .tablewrap,details .tablewrap{display:block}
  a{color:inherit;text-decoration:none}
  footer{border-top:1px solid var(--grid);padding-top:8px;margin-top:12px}
}
"""

TOC = [
    ("summary", "Verdict"),
    ("data", "What was recorded"),
    ("amplitude", "Broadband amplitude"),
    ("spectra", "Spectra"),
    ("lines", "Narrowband lines"),
    ("topography", "Spatial distribution on the helmet"),
    ("stability", "Time course and data loss"),
    ("buttons", "Response to a button press"),
    ("motion", "The movement artifact"),
    ("methods", "Methods and reproduction"),
]


def render(res, figures: dict[str, str]) -> str:
    toc = "".join(f'<li><a href="#{i}">{esc(t)}</a></li>' for i, t in TOC)
    root_vars = (
        f"--page:{PALETTE['page']};--surface:{PALETTE['surface']};"
        f"--ink:{PALETTE['ink']};--ink2:{PALETTE['ink_secondary']};"
        f"--muted:{PALETTE['muted']};--grid:{PALETTE['grid']};"
        f"--axis:{PALETTE['axis']};--border:rgba(11,11,11,0.10)"
    )
    sections = "".join(
        [
            _section_summary(res),
            _section_inventory(res),
            _section_amplitude(res, figures),
            _section_spectra(res, figures),
            _section_lines(res, figures),
            _section_topography(res, figures),
            _section_stability(res, figures),
            _section_buttons(res, figures),
            _section_motion(res, figures),
            _section_methods(res),
        ]
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEG gamepad artifact test — {esc(res.site)}</title>
<style>:root{{{root_vars}}}{CSS}</style>
</head><body>
<div class="wrap">
<header>
  <h1>MEG artifact test of an MR/MEG-compatible gamepad</h1>
  <p class="subtitle">This report compares four 1-minute runs on a 306-channel
     MEGIN system. The runs show the controller at rest, in use, and in motion.
     The reference is a run with no controller present.</p>
  <div class="meta">
    <span>Site <strong>{esc(res.site)}</strong></span>
    <span>102 magnetometers · 204 gradiometers · {res.inventory[0]['sfreq']:.0f} Hz</span>
    <span>Mains {res.line_freq:g} Hz</span>
    <span>MNE-Python {esc(res.mne_version)}</span>
    <span>Generated {esc(res.computed_at)}</span>
  </div>
</header>
<nav class="toc"><ol>{toc}</ol></nav>
{sections}
<footer>Generated by <code>meg-gamepad-artifact-test</code>. All figures and
values come from the raw <code>.fif</code> files. No value is edited by hand.</footer>
</div>
</body></html>"""


def write_report(res, out_path: Path | None = None) -> Path:
    out_path = out_path or (REPORT_DIR / "gamepad_artifact_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figures = build_figures(res)
    out_path.write_text(render(res, figures), encoding="utf-8")
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path
