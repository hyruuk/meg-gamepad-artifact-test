"""Export the HTML report to PDF.

There is no pure-Python renderer that handles this page well, so the exporter
drives whichever external renderer the machine already has. Chrome is tried
first because it renders the page exactly as the browser does, which keeps the
two formats identical.

One quirk needs care. Headless Chrome writes the PDF correctly and then, in
several versions, does not exit. The exporter therefore watches the output file
instead of waiting for the process: when the file exists and stops growing, the
export is complete and the process is terminated.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Chrome-family executables, by the names and locations they actually use.
CHROME_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
)

INSTALL_HINT = (
    "No PDF renderer found. Install any one of: Google Chrome or Chromium "
    "(no configuration needed), weasyprint (pip install weasyprint), or "
    "wkhtmltopdf. Then run the analysis again, or pass --no-pdf to skip it."
)


class PdfError(RuntimeError):
    """No renderer was available, or the renderer produced no usable file."""


@dataclass
class Renderer:
    name: str
    executable: str


def _which(candidate: str) -> str | None:
    if "/" in candidate:
        return candidate if Path(candidate).is_file() else None
    return shutil.which(candidate)


def find_renderer() -> Renderer | None:
    """First available renderer, in order of output quality."""
    for candidate in CHROME_CANDIDATES:
        path = _which(candidate)
        if path:
            return Renderer("chrome", path)
    for name in ("weasyprint", "wkhtmltopdf"):
        path = shutil.which(name)
        if path:
            return Renderer(name, path)
    return None


def _valid_pdf(path: Path) -> bool:
    """A complete PDF starts with the magic bytes and ends with its trailer."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False
        fh.seek(max(0, path.stat().st_size - 4096))
        return b"%%EOF" in fh.read()


def _run_watching_output(
    cmd: list[str], out_path: Path, timeout: float, settle: float = 1.5
) -> None:
    """Run `cmd`, and stop as soon as `out_path` is written and stable.

    This is what makes the Chrome path reliable: the renderer may never exit,
    but the file it produces is finished once its size stops changing.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    deadline = time.time() + timeout
    last_size, stable_since = -1, None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            if out_path.exists():
                size = out_path.stat().st_size
                if size > 0 and size == last_size:
                    stable_since = stable_since or time.time()
                    if time.time() - stable_since >= settle:
                        log.debug("%s finished writing; stopping renderer", out_path)
                        return
                else:
                    stable_since = None
                last_size = size
            time.sleep(0.25)
        raise PdfError(
            f"The PDF renderer did not finish within {timeout:.0f} s."
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def html_to_pdf(
    html_path: Path, pdf_path: Path | None = None, timeout: float = 180.0
) -> Path:
    """Render an HTML report to PDF with whichever renderer is installed."""
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise PdfError(f"{html_path} does not exist.")
    pdf_path = Path(pdf_path) if pdf_path else html_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()

    renderer = find_renderer()
    if renderer is None:
        raise PdfError(INSTALL_HINT)

    log.info("rendering PDF with %s", renderer.name)
    if renderer.name == "chrome":
        # A throwaway profile, so the export never touches the user's own
        # Chrome session or competes with a running instance for its profile.
        profile = tempfile.mkdtemp(prefix="meg-gamepad-chrome-")
        try:
            _run_watching_output(
                [
                    renderer.executable,
                    "--headless",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--no-pdf-header-footer",
                    f"--user-data-dir={profile}",
                    f"--print-to-pdf={pdf_path}",
                    html_path.as_uri(),
                ],
                pdf_path,
                timeout,
            )
        finally:
            shutil.rmtree(profile, ignore_errors=True)
    else:
        cmd = (
            [renderer.executable, str(html_path), str(pdf_path)]
            if renderer.name == "weasyprint"
            else [
                renderer.executable,
                "--enable-local-file-access",
                str(html_path),
                str(pdf_path),
            ]
        )
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            raise PdfError(
                f"{renderer.name} failed: "
                f"{result.stderr.decode('utf-8', 'replace')[:400]}"
            )

    if not _valid_pdf(pdf_path):
        raise PdfError(
            f"{renderer.name} did not produce a usable PDF at {pdf_path}."
        )
    log.info("wrote %s (%.1f MB)", pdf_path, pdf_path.stat().st_size / 1e6)
    return pdf_path
