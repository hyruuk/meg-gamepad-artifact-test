# MEG gamepad artifact test

Does an MR/MEG-compatible gamepad put artifacts into the MEG?

The reference dataset is **`tomov_mgh`**: four one-minute runs on a 306-channel
MEGIN system (102 magnetometers, 204 planar gradiometers, 1 kHz, 0.1–330 Hz)
with the same participant in the helmet:

| Run | What the participant was doing |
|---|---|
| `baseline` | In the helmet, **no controller present** |
| `passive`  | Holding the powered controller still, hands at hips |
| `playing`  | Holding the controller still and **pressing buttons** |
| `motion`   | **Cyclically raising the controller to the face** and back, pressing a button at the face |

This repo turns those four files into a single self-contained HTML report that
quantifies how much artifact the device adds, of what kind, and where it sits in
the spectrum.

## Findings

**The controller is electromagnetically quiet.** Powered and held still — idle or
with buttons being pressed — the data is indistinguishable from the
no-controller baseline on every measure computed:

- **Above 45 Hz:** +0.42 dB at worst, across every condition. No broadband noise.
- **No new spectral lines.** No narrowband peak above 30 Hz gained the 3 dB of
  prominence that would mark a clock, polling rate, or switching regulator. The
  closest was 49.1 Hz at +2.8 dB.
- **No extra data loss.** At conventional peak-to-peak thresholds nothing is
  rejected in any condition. The button-pressing run in fact has the *lowest*
  median peak-to-peak amplitude of the four, on both sensor types.
- **No button-press artifact.** Averaging 59 presses shows nothing locked to the
  TTL once slow drift is removed, bounding any switch or trigger-crosstalk
  transient below **185 fT** on magnetometers and **102 fT/cm** on gradiometers.
  The presses are also not phase-locked to the mains.

**Moving the controller produces a real movement artifact**, and only there:
**+4.0 dB** of sub-1 Hz drift and **+3.25 dB** in the 1–45 Hz band relative to
holding the same controller still — but the excess stays above 3 dB only up to
about **2.3 Hz**, and above 45 Hz it is **+0.03 dB**. Cycle-locked peak-to-peak
amplitude is ~1.2 pT on magnetometers, strongest over the lateral sensors the
hands pass closest to. This is an ordinary low-frequency movement artifact, in
the band standard high-pass filtering already handles — not interference from
the device.

**Two findings that look like the device but are not:**

- *Mains harmonics* (180 / 240 / 300 Hz) grew in the two static controller runs.
  They gain **11.1 dB on magnetometers but only 2.3 dB on gradiometers** —
  magnetometers see distant near-uniform fields that planar gradiometers reject,
  so the source is far away, not in the participant's hands. They are also absent
  from the moving run, which had the same controller powered on.
- *Alpha power drops* 4.2 dB while the participant presses buttons. That is
  alpha desynchronisation during an engaging task — physiology, and a useful
  positive control that the pipeline is sensitive enough to see real effects.

## What the dataset cannot settle

Worth reading before quoting any number above:

- The runs are **sequential, not interleaved** — the baseline was recorded
  12 minutes before the controller runs, so environmental drift over those
  minutes masquerades as a controller effect. (The mains-harmonic finding is
  exactly this trap; the report resolves it with a sensor-geometry argument.)
  For this reason the moving run is graded against the passive run, not the
  baseline.
- **No empty-room run.** Every run contains a participant, so some differences
  between conditions are the participant rather than the hardware.
- **No head-position tracking.** These files carry no HPI data and no
  device-to-head transform, so the movement artifact cannot be separated into
  "head moved" versus "controller moved".
- **One participant, one minute per condition.** Enough to detect a gross
  artifact, not to bound a small one precisely.

A follow-up that would close the main gap: move the controller through the empty
helmet with no participant, which isolates the device's own magnetic signature
from the body movement that carries it.

## Running it

```bash
pip install -e .            # or: pip install mne numpy scipy matplotlib
python scripts/run_analysis.py          # or: meg-gamepad
open reports/tomov_mgh.html             # and reports/tomov_mgh.pdf
```

Each run writes the report twice: `reports/<site>.html`, which is
self-contained and needs no other files, and `reports/<site>.pdf` for
circulation and archiving. The PDF is the same document paginated for A4, with
figures, tables and verdict cards kept whole and table headers repeated across
pages.

The PDF step drives whichever renderer the machine has — Chrome or Chromium
first, then weasyprint or wkhtmltopdf. If none is installed the analysis still
completes and writes the HTML, and prints what to install. Pass `--no-pdf` to
skip the step.

A full run takes about half a minute. `python scripts/run_analysis.py --help`
lists the options; `--cache` stores intermediate metrics, which is only worth it
when iterating on report text alone.

### The data

The `.fif` files are **not in git** — they are ~300 MB of raw MEG. Put them in
`data/tomov_mgh/`, beside the `dataset.json` that is tracked here. The report is
written to `reports/<site>.html`.

## Running it at another site

Nothing about a site is hard-coded. A dataset is a directory of recordings plus
a `dataset.json` that says which file is which run:

```bash
mkdir -p data/yoursite && cp /path/to/*.fif data/yoursite/
meg-gamepad --data-dir data/yoursite --init-manifest   # writes a template
$EDITOR data/yoursite/dataset.json                     # name the four runs
meg-gamepad --site yoursite                            # -> reports/yoursite.html
```

```json
{
  "site": "yoursite",
  "runs": {
    "baseline": "whatever_you_called_it.fif",
    "passive":  "...",
    "playing":  "...",
    "motion":   "..."
  },
  "button_codes": { "playing": [512, 2048], "motion": [2048] }
}
```

`runs` carries the file names, so any naming convention works. `button_codes`
carries the STI101 bit values your buttons raise, which depend on local trigger
wiring — check them against your own trigger channel before trusting the
event-locked sections. Mains frequency is read from the file headers, and the
band limits above 30 Hz follow from it, so a 50 Hz site gets 30–48 Hz and
52–100 Hz gamma bands instead of 30–58 and 62–100. Add `"line_freq": 50` to the
manifest to override it.

Without a manifest the loader guesses the runs from the file names and says so
loudly in the log. It refuses to guess when two files match the same run, rather
than risk mislabelling a condition and inverting the conclusions.

## What the report contains

1. **Verdict** — a graded card per condition against a published threshold table.
2. **What was recorded** — acquisition inventory, a matched-acquisition check,
   and the caveats above.
3. **Broadband amplitude** — per-channel RMS in three passbands. Separated by
   passband because magnetometer RMS is otherwise dominated by sub-1 Hz drift.
4. **Spectra** — absolute PSD, change vs baseline in dB, and a per-band table.
5. **Narrowband lines** — automatic peak detection with prominence over the local
   spectral floor, classified into powerline harmonics, the physiological band,
   and genuine instrument candidates; plus the magnetometer-vs-gradiometer test
   that locates the harmonic source.
6. **Where on the helmet** — topography of the power change per band.
7. **Time course and data loss** — sliding-window amplitude, and epochs lost
   across the whole range of rejection thresholds rather than at one arbitrary cut.
8. **Button presses** — event-locked averages tested against a random-event null
   on three statistics: any locked deflection, whether it survives a 2 Hz
   high-pass, and 60 Hz phase-locking.
9. **Movement artifact** — cycle-locked fields, topography, and the
   moved-vs-held-still contrast.
10. **Methods** — every parameter, and how to reproduce.

## Layout

```
data/<site>/    recordings (gitignored) + dataset.json (tracked)
src/meg_gamepad/
  config.py     conditions, bands, thresholds, palette -- all tunables live here
  pdf.py        HTML -> PDF via whichever renderer is installed
  dataset.py    per-site manifest: file names, trigger codes, mains frequency
  loading.py    reading .fif, cropping to a common window, decoding button events
  metrics.py    spectra, band powers, line detection, amplitude, rejection
  events.py     button-locked and movement-cycle analyses, surrogate tests
  figures.py    all plots, rendered to base64 PNG
  report.py     HTML assembly
  cli.py        command-line entry point
scripts/run_analysis.py    runs the whole thing without installing
```

Analysis is [MNE-Python](https://mne.tools) throughout.
