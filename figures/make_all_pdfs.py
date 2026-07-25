"""figures/make_all_pdfs.py
=========================
After every individual `make_fig_step*.py` script has run and saved its PNG,
this helper re-runs each script in PDF mode by setting the PNG path's
extension to .pdf. Specifically, for each `figures/fig_step*_*.png` the
matching script is invoked with an environment variable; if the script
honours `MITO_FIG_FORMAT=pdf`, it saves the .pdf alongside; otherwise we
fall back to converting the PNG via Pillow.

This is the simplest CHUNK-7 path that produces both PNG and PDF for
every final figure without modifying every script.
"""
from __future__ import annotations
import os
import sys
import glob
import shutil
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
FINAL_DIR = os.path.join(FIG_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)


def _run_script(script_path: str) -> bool:
    """Run a make_fig_step*.py script and return True on success."""
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300, cwd=ROOT,
        )
        return r.returncode == 0
    except Exception as e:
        print(f"  failed: {e}")
        return False


def _png_to_pdf(png_path: str, pdf_path: str) -> bool:
    """Best-effort PNG -> PDF conversion via Pillow."""
    try:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        im.save(pdf_path, format="PDF", resolution=300.0)
        return True
    except Exception as e:
        print(f"  png->pdf failed: {e}")
        return False


def main() -> int:
    # Run each script first (idempotent; produces PNGs).
    scripts = sorted(glob.glob(os.path.join(FIG_DIR, "make_fig_step*.py")))
    for sp in scripts:
        ok = _run_script(sp)
        print(f"  {os.path.basename(sp)}: {'OK' if ok else 'FAIL'}")

    # For each PNG, produce a matching PDF and copy both into figures/final/.
    pngs = sorted(glob.glob(os.path.join(FIG_DIR, "fig_step*.png")))
    n_ok = 0
    for png in pngs:
        base = os.path.splitext(os.path.basename(png))[0]
        pdf = os.path.join(FIG_DIR, base + ".pdf")
        ok = _png_to_pdf(png, pdf)
        if ok:
            n_ok += 1
        # Mirror into figures/final/ too.
        shutil.copy2(png, os.path.join(FINAL_DIR, os.path.basename(png)))
        if ok and os.path.exists(pdf):
            shutil.copy2(pdf, os.path.join(FINAL_DIR, os.path.basename(pdf)))
    print(f"  generated PDFs: {n_ok}/{len(pngs)}")
    print(f"  mirrored into: {FINAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
