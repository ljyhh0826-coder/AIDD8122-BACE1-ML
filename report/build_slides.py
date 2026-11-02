"""Build the 10-minute presentation deck (PowerPoint, then PDF).

The deck follows the report but is paced for ten minutes: one idea per slide, the figure doing the
work and the text reduced to the claim it supports.

Usage: python report/build_slides.py [--name "..."] [--github URL]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGURES = ROOT / "figures"
OUT_PPTX = ROOT / "report" / "AIDD8122_Final_Project_Slides.pptx"

from report import content as C  # noqa: E402

W, H = Cm(33.87), Cm(19.05)  # 16:9
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
ACCENT2 = RGBColor(0xEB, 0x68, 0x34)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
CARD = RGBColor(0xF6, 0xF5, 0xF1)

MARGIN = Cm(1.6)
TITLE_TOP = Cm(0.9)
BODY_TOP = Cm(3.3)
DASH = "—"
ARROW = "→"


def textbox(slide, left, top, width, height, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
            spacing=1.25, anchor=MSO_ANCHOR.TOP):
    """A text box whose content is a string or a list of lines; a line may be (marker, text)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(6)
        marker, body = line if isinstance(line, tuple) else (None, line)
        if marker:
            run = p.add_run()
            run.text = marker + "  "
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = ACCENT
            run.font.name = "Arial"
        run = p.add_run()
        run.text = body
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Arial"
    return box


def add_rule(slide, top):
    line = slide.shapes.add_shape(1, MARGIN, top, W - 2 * MARGIN, Pt(1))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False
    return line


def slide_base(prs, title, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    if kicker:
        textbox(slide, MARGIN, TITLE_TOP, W - 2 * MARGIN, Cm(0.9), kicker.upper(), size=11, color=MUTED)
        textbox(slide, MARGIN, TITLE_TOP + Cm(0.75), W - 2 * MARGIN, Cm(1.4), title, size=26, bold=True)
        add_rule(slide, TITLE_TOP + Cm(2.1))
    else:
        textbox(slide, MARGIN, TITLE_TOP + Cm(0.2), W - 2 * MARGIN, Cm(1.4), title, size=26, bold=True)
        add_rule(slide, TITLE_TOP + Cm(1.9))
    return slide


def add_figure(slide, name, top, max_h, left=None, max_w=None):
    """Place a figure scaled to fit the given box, centred horizontally unless `left` is given."""
    path = FIGURES / name
    if not path.exists():
        return None
    w_px, h_px = Image.open(path).size
    max_w = max_w or (W - 2 * MARGIN)
    scale = min(max_w / w_px, max_h / h_px)
    w, h = int(w_px * scale), int(h_px * scale)
    left = left if left is not None else int((W - w) / 2)
    return slide.shapes.add_picture(str(path), left, top, width=w, height=h)


def stat_tiles(slide, items, top, height=Cm(3.3)):
    """A row of headline numbers. Labels may hold a newline, so the card fits two lines."""
    n = len(items)
    gap = Cm(0.5)
    width = int((W - 2 * MARGIN - gap * (n - 1)) / n)
    for i, (value, label, colour) in enumerate(items):
        left = MARGIN + i * (width + gap)
        card = slide.shapes.add_shape(5, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD
        card.line.fill.background()
        card.shadow.inherit = False
        textbox(slide, left, top + Cm(0.35), width, Cm(1.3), value, size=28, bold=True, color=colour,
                align=PP_ALIGN.CENTER)
        textbox(slide, left + Cm(0.3), top + Cm(1.75), width - Cm(0.6), Cm(1.4), label.split("\n"),
                size=12, color=INK2, align=PP_ALIGN.CENTER, spacing=1.05)


def build(summary, student, github):
    d, sp, b = summary["data"], summary["splits"], summary["benchmark"]
    gap, hpo = summary["split_gap"], summary.get("hpo", {})
    gnn, ens = summary.get("gnn", {}), summary.get("ensemble", {})
    ad = summary.get("interpretation", {}).get("applicability", {})
    cls_best, reg_best = b["cls"]["best"], b["reg"]["best"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(s, MARGIN, Cm(4.2), W - 2 * MARGIN, Cm(1.0),
            f"AIDD8122 {DASH} Advanced Topics in Artificial Intelligence & Drug Discovery",
            size=13, color=MUTED, align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Cm(5.6), W - 2 * MARGIN, Cm(4.0),
            ["Molecular representations, model families and",
             "evaluation protocols for predicting BACE-1 inhibition"],
            size=30, bold=True, align=PP_ALIGN.CENTER, spacing=1.15)
    textbox(s, MARGIN, Cm(9.9), W - 2 * MARGIN, Cm(1.0),
            "A systematic benchmark against graph neural networks",
            size=17, color=ACCENT, align=PP_ALIGN.CENTER)
    add_rule(s, Cm(11.6))
    textbox(s, MARGIN, Cm(12.3), W - 2 * MARGIN, Cm(2.0),
            [student, "Faculty of Applied Sciences · Macao Polytechnic University"],
            size=13, color=INK2, align=PP_ALIGN.CENTER, spacing=1.3)

    # 2 motivation
    s = slide_base(prs, "BACE-1: a well-studied target that keeps failing", "Motivation")
    textbox(s, MARGIN, BODY_TOP, W - 2 * MARGIN, Cm(4.6),
            [(DASH, "Alzheimer's disease: 55 million people, no reliable disease-modifying therapy. "
                    "BACE-1 performs the first, rate-limiting cut that releases amyloid-beta."),
             (DASH, "Four clinical candidates reached late-stage trials; all were discontinued "
                    f"{DASH} but two decades of chemistry left a rich, well-annotated dataset."),
             (ARROW, "An ideal testbed for asking what actually drives predictive accuracy.")],
            size=17, color=INK2, spacing=1.4)
    add_figure(s, "fig2_dataset.png", BODY_TOP + Cm(4.9), Cm(7.4))
    textbox(s, MARGIN, BODY_TOP + Cm(10.3), W - 2 * MARGIN, Cm(1.2),
            f"{d['n']:,} curated molecules · {d['n_active']} active · "
            f"{d['n_scaffolds']} Bemis-Murcko scaffolds · median MW {d['mw_median']:.0f} Da",
            size=13, color=MUTED, align=PP_ALIGN.CENTER)

    # 3 research questions
    s = slide_base(prs, "Three choices decide what a model appears to be worth", "Research questions")
    textbox(s, MARGIN, BODY_TOP, W - 2 * MARGIN, Cm(1.2),
            f"Every QSAR study makes these choices {DASH} usually implicitly. This project makes them "
            "the object of study.", size=16, color=INK2)
    cards = [("RQ1", "Representation vs model", "Which choice buys more accuracy for the same effort?"),
             ("RQ2", "Random vs scaffold split", "How optimistic is a random split, in numbers?"),
             ("RQ3", "Learned vs fixed features", "Does a graph network beat fingerprints at this scale?"),
             ("RQ4", "Accuracy vs reliability", "Can we say where the model should be trusted?")]
    gapx = Cm(0.55)
    cw = int((W - 2 * MARGIN - 3 * gapx) / 4)
    for i, (tag, head, body) in enumerate(cards):
        left = MARGIN + i * (cw + gapx)
        card = s.shapes.add_shape(5, left, BODY_TOP + Cm(2.0), cw, Cm(8.4))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD
        card.line.fill.background()
        card.shadow.inherit = False
        textbox(s, left + Cm(0.6), BODY_TOP + Cm(2.5), cw - Cm(1.2), Cm(0.9), tag, size=15, bold=True,
                color=ACCENT)
        textbox(s, left + Cm(0.6), BODY_TOP + Cm(3.5), cw - Cm(1.2), Cm(2.2), head, size=18, bold=True,
                spacing=1.15)
        textbox(s, left + Cm(0.6), BODY_TOP + Cm(6.0), cw - Cm(1.2), Cm(4.0), body, size=14, color=INK2,
                spacing=1.3)

    # 4 methods / workflow
    s = slide_base(prs, "One dataset, one protocol, everything else varied", "Methods")
    add_figure(s, "fig1_workflow.png", BODY_TOP + Cm(0.2), Cm(9.2))
    textbox(s, MARGIN, BODY_TOP + Cm(9.8), W - 2 * MARGIN, Cm(2.4),
            [(DASH, "6 representations × 6 model families × 2 tasks × 2 splits × "
                    "5 seeds, plus a GINE graph network written from scratch in PyTorch."),
             (DASH, "Every selection decision " + DASH + " hyperparameters, early stopping, ensemble "
                    "membership " + DASH + " made on validation data only.")],
            size=14, color=INK2, spacing=1.35)

    # 5 split design
    s = slide_base(prs, "Scaffold splitting genuinely hides chemistry from the model", "Evaluation design")
    add_figure(s, "fig3_splits.png", BODY_TOP + Cm(0.3), Cm(9.0))
    stat_tiles(s, [(f"{sp['random']['mean_max_sim']:.2f}",
                    "mean max Tanimoto similarity,\nrandom split", ACCENT),
                   (f"{sp['scaffold']['mean_max_sim']:.2f}",
                    "mean max Tanimoto similarity,\nscaffold split", ACCENT2),
                   (f"{d['n_scaffolds']}", "distinct Bemis-Murcko\nscaffolds", INK)],
                BODY_TOP + Cm(10.1))

    # 6 RQ1
    s = slide_base(prs, "Representation matters at least as much as the model", "Result · RQ1")
    add_figure(s, "fig4_benchmark.png", BODY_TOP + Cm(0.4), Cm(8.0))
    textbox(s, MARGIN, BODY_TOP + Cm(8.8), W - 2 * MARGIN, Cm(3.0),
            [(DASH, f"Spread between best and worst representation: {b['cls']['feature_spread']:.3f} "
                    f"ROC-AUC. Between best and worst model family: {b['cls']['model_spread']:.3f}."),
             (DASH, f"Best classifier {cls_best['name']} ({cls_best['mean']:.3f}); best regressor "
                    f"{reg_best['name']} ({reg_best['mean']:.3f} RMSE)."),
             (DASH, "Naive fusion of fingerprints and descriptors did not help " + DASH +
                    " it hurt the linear model.")],
            size=15, color=INK2, spacing=1.35)

    # 7 RQ2
    s = slide_base(prs, "A random split flatters every single model", "Result · RQ2")
    add_figure(s, "fig5_split_gap.png", BODY_TOP + Cm(0.1), Cm(9.2))
    stat_tiles(s, [(f"{gap['cls']['n_better_on_random']}/{gap['cls']['n_total']}",
                    "configurations scored better\nunder random splitting", ACCENT2),
                   (f"+{gap['cls']['mean']:.3f}", "mean ROC-AUC\ninflation", ACCENT2),
                   (f"−{gap['reg']['mean']:.3f}", "mean RMSE understatement\n(log units)", ACCENT2)],
                BODY_TOP + Cm(10.0))

    # 8 RQ3 graph neural network
    if gnn.get("cls"):
        base_c, base_r = gnn["cls"]["GNN (base)"], gnn["reg"]["GNN (base)"]
        s = slide_base(prs, "The graph network does not win at 1,200 molecules", "Result · RQ3")
        add_figure(s, "fig8_ablation.png", BODY_TOP + Cm(0.3), Cm(8.6))
        textbox(s, MARGIN, BODY_TOP + Cm(9.2), W - 2 * MARGIN, Cm(3.2),
                [(DASH, f"GNN reaches {base_c['mean']:.3f} ROC-AUC and {base_r['mean']:.3f} RMSE "
                        f"{DASH} behind the best fingerprint models ({cls_best['mean']:.3f} / "
                        f"{reg_best['mean']:.3f})."),
                 (DASH, "Ablations behave as theory predicts: depth peaks at three layers "
                        "(over-smoothing beyond), concatenated readout beats any single pooling."),
                 (DASH, f"Removing stereochemistry costs "
                        f"{base_c['mean'] - gnn['cls']['no chirality']['mean']:.3f} ROC-AUC "
                        f"{DASH} chemically sensible for transition-state isosteres.")],
                size=15, color=INK2, spacing=1.35)

    # 9 tuning and ensembling
    s = slide_base(prs, "Tuning buys little; ensembling helps only regression", "Result · RQ4a")
    add_figure(s, "fig6_tuning.png", BODY_TOP + Cm(0.3), Cm(8.0))
    textbox(s, MARGIN, BODY_TOP + Cm(8.8), W - 2 * MARGIN, Cm(3.0),
            [(DASH, f"20 random configurations per model, task and seed: mean change "
                    f"{hpo.get('cls', {}).get('mean_delta', 0):+.4f} ROC-AUC, only "
                    f"{hpo.get('cls', {}).get('n_improved', 0)} of 6 families improved."),
             (DASH, "With ~150 validation molecules, the best of twenty captures noise as readily as "
                    "signal: validation keeps rising, test does not follow."),
             (DASH, f"Ensembling: no gain for classification, but the best overall regression result "
                    f"({ens.get('reg', {}).get('mean', 0):.3f} RMSE).")],
            size=15, color=INK2, spacing=1.35)

    # 10 interpretation
    s = slide_base(prs, "The model's reasoning is chemically legible", "Result · RQ4b")
    add_figure(s, "fig9_shap.png", BODY_TOP + Cm(0.3), Cm(8.4))
    textbox(s, MARGIN, BODY_TOP + Cm(9.0), W - 2 * MARGIN, Cm(2.4),
            [(DASH, "SHAP over ECFP4 bits maps each contribution back to a concrete substructure: "
                    "fluorinated and aromatic-nitrogen environments raise predicted activity."),
             (DASH, "Presence of a substructure is informative; absence is not " + DASH +
                    " a direct consequence of how fingerprints encode molecules.")],
            size=15, color=INK2, spacing=1.35)

    # 11 applicability domain, the practical punchline
    s = slide_base(prs, "A single accuracy number hides a twofold difference in reliability",
                   "Result · RQ4c — the practical finding")
    add_figure(s, "fig11_applicability.png", BODY_TOP + Cm(0.3), Cm(7.8))
    acc_lo = ad.get("accuracy_by_bin", {}).get("[0.0, 0.4)", 0) * 100
    acc_hi = ad.get("accuracy_by_bin", {}).get("[0.8, 1.01)", 0) * 100
    rmse_lo = ad.get("rmse_by_bin", {}).get("[0.0, 0.4)", 0)
    rmse_hi = ad.get("rmse_by_bin", {}).get("[0.8, 1.01)", 0)
    stat_tiles(s, [(f"{acc_lo:.0f}% {ARROW} {acc_hi:.0f}%",
                    "accuracy, least to most similar\nto the training set", ACCENT),
                   (f"{rmse_lo:.2f} {ARROW} {rmse_hi:.2f}", "pIC50 RMSE over the\nsame range", ACCENT),
                   (f"{ad.get('acc_top10pct', 0) * 100:.0f}%",
                    "accuracy on the most\nconfident 10%", ACCENT2)],
                BODY_TOP + Cm(8.7))

    # 12 final comparison
    s = slide_base(prs, "Everything side by side", "Summary of results")
    add_figure(s, "fig12_final.png", BODY_TOP + Cm(0.4), Cm(9.4))
    textbox(s, MARGIN, BODY_TOP + Cm(10.2), W - 2 * MARGIN, Cm(1.6),
            "Scaffold-split test performance, mean ± SD over five seeds. Blue: classical models. "
            "Green: graph networks. Orange: ensemble.", size=14, color=MUTED, align=PP_ALIGN.CENTER)

    # 13 conclusions
    s = slide_base(prs, "What to take away", "Conclusions")
    textbox(s, MARGIN, BODY_TOP + Cm(0.4), W - 2 * MARGIN, Cm(10),
            [("1", "Spend effort on the representation before the model " + DASH +
                   " it explains as much or more of the variation."),
             ("2", f"Report scaffold-split numbers. A random split inflated every configuration here, "
                   f"by {gap['cls']['mean']:.3f} ROC-AUC on average."),
             ("3", "At ~1,200 molecules a learned graph representation does not beat well-designed "
                   "fingerprints; descriptors recover part of its gap."),
             ("4", "Tuning and ensembling give little on a small validation set " + DASH +
                   " the noise floor is reached quickly."),
             ("5", "Always publish an applicability domain: reliability varied twofold with similarity "
                   "to the training set, and confidence tracked it closely enough to triage.")],
            size=17, color=INK2, spacing=1.5)

    # 14 closing
    s = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(s, MARGIN, Cm(6.0), W - 2 * MARGIN, Cm(1.6), "Thank you", size=34, bold=True,
            align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Cm(8.2), W - 2 * MARGIN, Cm(1.2), "Questions welcome", size=18, color=INK2,
            align=PP_ALIGN.CENTER)
    add_rule(s, Cm(10.4))
    textbox(s, MARGIN, Cm(11.1), W - 2 * MARGIN, Cm(2.4),
            ["Code, cached splits, per-molecule predictions and figure scripts:", github],
            size=14, color=ACCENT, align=PP_ALIGN.CENTER, spacing=1.4)
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[Student Name]  ·  [P00000]")
    ap.add_argument("--github", default=None)
    args = ap.parse_args()
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    prs = build(summary, args.name, args.github or C.GITHUB_URL)
    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"wrote {OUT_PPTX} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
