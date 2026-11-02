"""Build the final report by filling the MPU Word template with text, figures and tables.

Usage: python report/build_report.py [--name "..."] [--id P12345] [--github URL]
"""
import argparse
import copy
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT.parent.parent / "AIDD8122_Final_Project_Report_Template.docx"
OUT_DOCX = ROOT / "report" / "AIDD8122_Final_Project_Report.docx"
FIGURES = ROOT / "figures"
TABLES = ROOT / "results" / "tables"

from report import content as C  # noqa: E402

FIGURE_ORDER = ["fig1_workflow.png", "fig2_dataset.png", "fig3_splits.png", "fig4_benchmark.png",
                "fig5_split_gap.png", "fig6_tuning.png", "fig7_roc.png", "fig8_ablation.png",
                "fig9_shap.png", "fig10_atoms.png", "fig11_applicability.png", "fig12_final.png"]
TABLE_ORDER = ["table_benchmark_cls.csv", "table_benchmark_reg.csv", "table_tuning_cls.csv",
               "table_tuning_reg.csv", "table_gnn_cls.csv", "table_gnn_reg.csv"]
# Where each figure and table is placed: after the paragraph whose text starts with this marker.
FIGURE_WIDTH = {"fig1_workflow.png": 16.0, "fig2_dataset.png": 16.0, "fig3_splits.png": 16.0,
                "fig4_benchmark.png": 16.0, "fig5_split_gap.png": 15.0, "fig6_tuning.png": 16.0,
                "fig7_roc.png": 14.0, "fig8_ablation.png": 16.0, "fig9_shap.png": 16.0,
                "fig10_atoms.png": 16.0, "fig11_applicability.png": 16.0, "fig12_final.png": 16.0}


def tidy(text):
    """Collapse the whitespace that indented triple-quoted strings leave behind.

    A hyphen immediately followed by a line break is a word the source wrapped mid-hyphenation
    ("descriptor-\\n based"), so it is rejoined; a spaced dash (" - ") is punctuation and is kept.
    """
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def thousands(n):
    return f"{int(n):,}"


class ReportBuilder:
    def __init__(self, template, summary):
        self.doc = Document(str(template))
        self.s = summary
        self.body = self.doc.element.body
        self.anchor = self._find_anchor()
        self.fig_no = 0
        self.tab_no = 0

    def _find_anchor(self):
        """Insert immediately before the References heading, which is the last block of the body.

        The template's table of contents also contains a "References" line, so the heading is
        identified by its Heading 1 style rather than by its text alone.
        """
        heading = self._references_heading()
        prev = heading.getprevious()
        # Keep the body text ahead of the page break that separates it from the references.
        if prev is not None and prev.tag == qn("w:p") and any(
                b.get(qn("w:type")) == "page" for b in prev.iter(qn("w:br"))):
            return prev
        return heading

    def _references_heading(self):
        found = None
        for el in self.body:
            if el.tag != qn("w:p") or "References" not in "".join(el.itertext()):
                continue
            style = el.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
            if style is not None and style.get(qn("w:val")) == "2":
                found = el
        if found is None:
            raise RuntimeError("References heading not found in template")
        return found

    def _new_paragraph(self, style=None):
        p = self.doc.add_paragraph(style=style)
        self.anchor.addprevious(p._p)
        return p

    def heading(self, text, level):
        p = self._new_paragraph(style=f"Heading {level}")
        p.add_run(text)
        if level == 1:  # the template numbers Heading 1 as "§N."
            p.paragraph_format.space_before = Pt(14)
        return p

    def para(self, text, justify=True, first_line_indent=False):
        p = self._new_paragraph()
        run = p.add_run(tidy(text))
        run.font.size = Pt(12)
        pf = p.paragraph_format
        pf.space_after = Pt(6)
        pf.line_spacing = 1.5
        if justify:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if first_line_indent:
            pf.first_line_indent = Cm(0.75)
        return p

    def caption(self, text, kind):
        p = self._new_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label = p.add_run(f"{kind} {self.fig_no if kind == 'Figure' else self.tab_no}. ")
        label.bold = True
        label.font.size = Pt(10)
        body = p.add_run(tidy(text))
        body.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(12)
        p.paragraph_format.space_before = Pt(4)
        return p

    def figure(self, name):
        path = FIGURES / name
        if not path.exists():
            return
        self.fig_no += 1
        p = self._new_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=Cm(FIGURE_WIDTH.get(name, 15.5)))
        self.caption(C.FIGURE_CAPTIONS.get(name, name), "Figure")

    def table(self, name):
        path = TABLES / name
        if not path.exists():
            return
        self.tab_no += 1
        df = pd.read_csv(path)
        self.caption(C.TABLE_CAPTIONS.get(name, name), "Table")
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, col in zip(t.rows[0].cells, df.columns):
            run = cell.paragraphs[0].add_run(str(col).replace("_", " "))
            run.bold = True
            run.font.size = Pt(8.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for cell, v in zip(cells, row):
                run = cell.paragraphs[0].add_run(str(v))
                run.font.size = Pt(8.5)
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.anchor.addprevious(t._tbl)
        spacer = self._new_paragraph()
        spacer.paragraph_format.space_after = Pt(10)

    def cover(self, title, student_name, student_id, submission):
        table = self.doc.tables[0]
        self._set_cell(table.cell(0, 0), title, bold=True, size=14, center=True)
        for row in table.rows:
            label = row.cells[0].text.strip()
            if label.startswith("Student ID"):
                self._set_cell(row.cells[1], student_id)
            elif label.startswith("Student Name"):
                self._set_cell(row.cells[1], student_name)
            elif label.startswith("Submission Date"):
                self._set_cell(row.cells[1], submission)

    @staticmethod
    def _set_cell(cell, text, bold=False, size=12, center=False):
        cell.text = ""
        p = cell.paragraphs[0]
        if center:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        run.font.name = "Times New Roman"

    def references(self, items):
        """Replace the template's empty bibliography placeholder with the real reference list."""
        for sdt in self.body.findall(qn("w:sdt")):
            self.body.remove(sdt)
        ref_p = self._references_heading()
        # The template suppresses the heading number on References; restore it so the section is
        # numbered §7 as the template's own table of contents expects.
        num_pr = ref_p.find(f"{qn('w:pPr')}/{qn('w:numPr')}")
        if num_pr is not None:
            ref_p.find(qn("w:pPr")).remove(num_pr)
        # The template puts an explicit page break before References, which leaves a blank page when
        # the body happens to end near a page boundary. Use "page break before" on the heading instead.
        prev = ref_p.getprevious()
        if prev is not None and prev.tag == qn("w:p") and any(
                b.get(qn("w:type")) == "page" for b in prev.iter(qn("w:br"))):
            self.body.remove(prev)
        pPr = ref_p.find(qn("w:pPr"))
        page_break_before = pPr.makeelement(qn("w:pageBreakBefore"), {})
        pPr.insert(0, page_break_before)
        cursor = ref_p
        for i, text in enumerate(items, 1):
            p = self.doc.add_paragraph()
            run = p.add_run(f"[{i}]\t{text}")
            run.font.size = Pt(10)
            pf = p.paragraph_format
            pf.left_indent = Cm(1.1)
            pf.first_line_indent = Cm(-1.1)
            pf.space_after = Pt(6)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            cursor.addnext(p._p)
            cursor = p._p

    def render(self, blocks):
        for block in blocks:
            kind = block[0]
            if kind == "h1":
                self.heading(block[1], 1)
            elif kind == "h2":
                self.heading(block[1], 2)
            elif kind == "p":
                self.para(block[1])
            elif kind == "figure":
                self.figure(block[1])
            elif kind == "table":
                self.table(block[1])
            elif kind == "gnn_results":
                self.render(C.gnn_results(self.s, thousands))
            elif kind == "ensemble":
                self.render(C.ensemble(self.s, thousands))
            elif kind == "conclusions":
                self.render(C.conclusions(self.s, thousands))
            else:
                raise ValueError(kind)


def place_assets(blocks):
    """Insert each figure and table after the paragraph that first refers to it."""
    out = []
    pending_fig = list(FIGURE_ORDER)
    pending_tab = list(TABLE_ORDER)
    fig_label = {name: f"Figure {i + 1}" for i, name in enumerate(FIGURE_ORDER)}
    tab_label = {name: f"Table {i + 1}" for i, name in enumerate(TABLE_ORDER)}
    for block in blocks:
        out.append(block)
        if block[0] != "p":
            continue
        text = block[1]
        for name in list(pending_fig):
            if fig_label[name] in text:
                out.append(("figure", name))
                pending_fig.remove(name)
        for name in list(pending_tab):
            if tab_label[name] in text:
                out.append(("table", name))
                pending_tab.remove(name)
    return out, pending_fig, pending_tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[Student Name]")
    ap.add_argument("--id", dest="sid", default="[P00000]")
    ap.add_argument("--github", default=None)
    ap.add_argument("--date", default=date.today().strftime("%d/%m/%Y"))
    args = ap.parse_args()
    if args.github:
        C.GITHUB_URL = args.github

    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    builder = ReportBuilder(TEMPLATE, summary)
    builder.cover(C.TITLE, args.name, args.sid, args.date)

    blocks = C.build(summary, thousands)
    # Expand the generated sections first so that their figure references are seen by place_assets.
    expanded = []
    for block in blocks:
        if block[0] == "gnn_results":
            expanded += C.gnn_results(summary, thousands)
        elif block[0] == "ensemble":
            expanded += C.ensemble(summary, thousands)
        elif block[0] == "conclusions":
            expanded += C.conclusions(summary, thousands)
        else:
            expanded.append(block)
    placed, missing_fig, missing_tab = place_assets(expanded)
    if missing_fig or missing_tab:
        print("WARNING: never referenced in the text:", missing_fig, missing_tab)

    builder.render(placed)
    builder.references(C.REFERENCES)
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    builder.doc.save(str(OUT_DOCX))
    print(f"wrote {OUT_DOCX}  ({builder.fig_no} figures, {builder.tab_no} tables, "
          f"{len(C.REFERENCES)} references)")


if __name__ == "__main__":
    main()
