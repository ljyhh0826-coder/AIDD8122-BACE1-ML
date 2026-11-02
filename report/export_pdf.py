"""Update the Word fields (table of contents, page numbers) and export the report to PDF.

Word is driven through COM, which is the only reliable way to refresh a TOC field on Windows.
A private Word instance is used so that a document the user already has open is never touched,
and calls are retried because Word rejects COM calls while it is busy.

Usage: python report/export_pdf.py [docx] [pdf]
"""
import sys
import time
from pathlib import Path

import pythoncom
from win32com.client import DispatchEx, gencache

WD_EXPORT_PDF = 17
STAT_WORDS, STAT_PAGES = 0, 2
RPC_BUSY = (-2147418111, -2147417846)  # "call rejected by callee", "application is busy"


def retry(fn, attempts=15, delay=1.5):
    """Retry while the Office application is starting up or busy serving another call.

    A busy application surfaces in several ways: the COM "call rejected" errors, an AttributeError
    when a member cannot be resolved yet, and a TypeError from gencache when the type library
    cannot be read because the server is not answering.
    """
    for i in range(attempts):
        try:
            return fn()
        except (pythoncom.com_error, AttributeError, TypeError) as exc:
            transient = not isinstance(exc, pythoncom.com_error) or exc.hresult in RPC_BUSY
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def export(docx, pdf):
    docx, pdf = Path(docx).resolve(), Path(pdf).resolve()
    gencache.EnsureModule("{00020905-0000-0000-C000-000000000046}", 0, 8, 7)  # Word type library
    word = DispatchEx("Word.Application")  # a private instance, not the user's running Word
    word.Visible = False
    word.DisplayAlerts = False
    doc = retry(lambda: word.Documents.Open(FileName=str(docx), ReadOnly=False, AddToRecentFiles=False))
    try:
        retry(doc.Fields.Update)
        for toc in doc.TablesOfContents:
            retry(toc.Update)
        retry(doc.Repaginate)
        pages = retry(lambda: doc.ComputeStatistics(STAT_PAGES))
        words = retry(lambda: doc.ComputeStatistics(STAT_WORDS))
        retry(lambda: doc.SaveAs2(str(pdf), WD_EXPORT_PDF))
        retry(doc.Save)
    finally:
        retry(lambda: doc.Close(False))
        retry(word.Quit)
    return pages, words


if __name__ == "__main__":
    here = Path(__file__).parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "AIDD8122_Final_Project_Report.docx"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    pages, words = export(src, dst)
    print(f"{dst.name}: {pages} pages, {words} words")
