"""Export the slide deck to PDF through a private PowerPoint instance.

Usage: python report/export_slides_pdf.py [pptx] [pdf]
"""
import sys
from pathlib import Path

from win32com.client import Dispatch, gencache

from export_pdf import retry  # same COM-busy retry policy

PP_EXPORT_PDF = 32
MS_FALSE = 0


def _powerpoint_running():
    try:
        import pythoncom
        pythoncom.GetActiveObject("PowerPoint.Application")
        return True
    except Exception:
        return False


def export(pptx, pdf):
    """PowerPoint is single-instance, so an already-running copy is reused and left running."""
    pptx, pdf = Path(pptx).resolve(), Path(pdf).resolve()
    was_running = _powerpoint_running()
    # EnsureDispatch builds the type library, without which Presentations is not resolvable; the
    # application also takes a moment to accept calls after it starts.
    app = retry(lambda: gencache.EnsureDispatch("PowerPoint.Application"))
    retry(lambda: app.Presentations)
    deck = retry(lambda: app.Presentations.Open(str(pptx), WithWindow=MS_FALSE))
    try:
        n = deck.Slides.Count
        retry(lambda: deck.SaveAs(str(pdf), PP_EXPORT_PDF))
    finally:
        retry(deck.Close)
        if not was_running:  # never close a copy the user already had open
            retry(app.Quit)
    return n


if __name__ == "__main__":
    here = Path(__file__).parent
    sys.path.insert(0, str(here))
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "AIDD8122_Final_Project_Slides.pptx"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    print(f"{dst.name}: {export(src, dst)} slides")
