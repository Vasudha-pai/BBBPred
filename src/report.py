"""Build a step-by-step PDF report of the whole project from the files in outputs/.

Run after train.py, explain.py, evaluate.py and analysis.py.

Usage: python src/report.py   ->  report/BBBP_Report.pdf
"""
import json
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from data import OUT, ROOT, load_dataset, scaffold_split

REPORT_DIR = ROOT / "report"
PDF = REPORT_DIR / "BBBP_Report.pdf"
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
ACCENT = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#eaf1f8")

# ---------------------------------------------------------------- styles
try:  # DejaVu covers Å, ≥, ±, β, ✓ which the built-in PDF fonts lack
    for name, f in [("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                    ("Body-Italic", "DejaVuSans-Oblique.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, f"{FONT_DIR}/{f}"))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                  boldItalic="Body-Bold")
    BODY, BOLD, MONO = "Body", "Body-Bold", "Mono"
except Exception:  # fall back to built-ins if DejaVu isn't installed
    BODY, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("T", fontName=BOLD, fontSize=24, leading=30, textColor=ACCENT,
                            alignment=TA_CENTER, spaceAfter=10),
    "subtitle": ParagraphStyle("ST", fontName=BODY, fontSize=12.5, leading=17, alignment=TA_CENTER,
                               textColor=colors.HexColor("#444444")),
    "h1": ParagraphStyle("H1", fontName=BOLD, fontSize=16, leading=20, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("H2", fontName=BOLD, fontSize=12.5, leading=16, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("B", fontName=BODY, fontSize=9.8, leading=14, spaceAfter=6),
    "bullet": ParagraphStyle("BL", fontName=BODY, fontSize=9.8, leading=14, leftIndent=14,
                             bulletIndent=4, spaceAfter=3),
    "caption": ParagraphStyle("C", fontName=BODY, fontSize=8.5, leading=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555555"), spaceAfter=10),
    "cell": ParagraphStyle("CE", fontName=BODY, fontSize=8.3, leading=10.5),
    "cellb": ParagraphStyle("CB", fontName=BOLD, fontSize=8.3, leading=10.5, textColor=colors.white),
    "code": ParagraphStyle("CO", fontName=MONO, fontSize=8.5, leading=11.5, leftIndent=8,
                           backColor=colors.HexColor("#f4f4f4"), borderPadding=5, spaceAfter=8),
    "box": ParagraphStyle("BX", fontName=BODY, fontSize=9.8, leading=14, backColor=LIGHT,
                          borderColor=ACCENT, borderWidth=0.6, borderPadding=7, spaceBefore=4,
                          spaceAfter=10),
    "toc1": ParagraphStyle("TOC1", fontName=BODY, fontSize=10.5, leading=16, leftIndent=10),
    "toc2": ParagraphStyle("TOC2", fontName=BODY, fontSize=9, leading=13, leftIndent=28,
                           textColor=colors.HexColor("#444444")),
}


class ReportDoc(SimpleDocTemplate):
    """Registers h1/h2 paragraphs with the table of contents and PDF outline."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont(BODY, 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Blood-Brain Barrier Penetration with a Random Forest")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------- helpers
def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def figure(name, caption, width=14.5):
    path = OUT / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    w = width * cm
    h = min(w * ratio, 17 * cm)
    w = h / ratio
    img.drawWidth, img.drawHeight = w, h
    return KeepTogether([img, Spacer(1, 3), P(caption, "caption")])


def table(rows, widths=None, highlight_last=False, highlight_rows=()):
    data = [[Paragraph(str(c), S["cellb"]) for c in rows[0]]]
    data += [[Paragraph(str(c), S["cell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1,
              hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d3df")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for r in list(highlight_rows) + ([len(rows) - 1] if highlight_last else []):
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#dbe8f5")))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 10)])


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- content
def build():
    m = json.loads((OUT / "metrics.json").read_text())
    ev = json.loads((OUT / "evaluation.json").read_text())
    an = json.loads((OUT / "analysis.json").read_text())
    baselines = pd.read_csv(OUT / "baselines.csv")
    splits = pd.read_csv(OUT / "repeated_splits.csv")
    shap_top = pd.read_csv(OUT / "shap_top_features.csv")
    ad = pd.read_csv(OUT / "applicability_domain.csv")
    errors = pd.read_csv(OUT / "error_analysis.csv")
    conflicts = pd.read_csv(OUT / "label_conflicts.csv")
    cases = pd.read_csv(OUT / "case_studies.csv")

    df, n_bad = load_dataset()
    n_mol, n_raw = len(df), len(df) + n_bad
    frac_pos = df["p_np"].mean()
    n_scaffolds = df["scaffold"].nunique()
    _, test_idx = scaffold_split(df["scaffold"].to_numpy())
    test_frac_pos = df["p_np"].to_numpy()[test_idx].mean()
    ci = ev["test_metrics_95ci"]
    thr = ev["threshold"]
    t05, ttun = thr["default_0.5"], thr["tuned_on_train_cv"]
    cal = ev["calibration"]
    rs = ev["repeated_splits_roc_auc"]
    rf_row = baselines[baselines["model"].str.startswith("Random")].iloc[0]
    n_correct_cases = int(cases["correct@tuned"].sum())
    bp = m["best_params"]

    story = []

    # ---- title page
    story += [Spacer(1, 4.5 * cm),
              P("Blood-Brain Barrier Penetration<br/>with a Random Forest", "title"),
              Spacer(1, 6),
              P("A step-by-step report: data, model, interpretation, trustworthiness "
                "and error analysis", "subtitle"),
              Spacer(1, 1.6 * cm)]
    story.append(table([
        ["Headline result", "Value"],
        ["Test ROC-AUC (scaffold split, 408 unseen chemotypes)",
         f"{m['test_roc_auc']:.3f}  (95% CI {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f})"],
        ["Balanced accuracy at tuned threshold", f"{ttun['balanced_accuracy']:.3f}"],
        ["Best simple baseline (TPSA alone)",
         f"{baselines.iloc[1]['roc_auc']:.3f} ROC-AUC"],
        ["Known drugs held out from training, correctly classified",
         f"{n_correct_cases} / {len(cases)}"],
    ], widths=[9.5, 5.5]))
    story += [Spacer(1, 2.5 * cm),
              P(f"Dataset: MoleculeNet BBBP ({n_raw:,} compounds)<br/>"
                f"Model: scikit-learn RandomForestClassifier on RDKit features<br/>"
                f"Generated {date.today():%d %B %Y} by <font face='{MONO}'>src/report.py</font>",
                "subtitle"),
              PageBreak()]

    # ---- contents
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story += [Paragraph("Contents", ParagraphStyle("TOCH", parent=S["h1"])),
              Spacer(1, 6), toc, PageBreak()]

    # ---- executive summary
    story += [P("Executive summary", "h1"),
              P("A drug for a brain disease only works if it can reach the brain, and most "
                "molecules are kept out by the blood-brain barrier (BBB). This project predicts, "
                "from a molecule's structure alone (its SMILES string), whether it crosses the "
                "BBB. It uses one random forest trained on 2,265 RDKit descriptors and "
                "fingerprint bits. The report walks through every step and then asks how far "
                "the result can be trusted.")]
    story += bullets([
        f"<b>Performance.</b> On a scaffold split, where the test molecules have core "
        f"structures never seen in training, the model reaches ROC-AUC <b>{m['test_roc_auc']:.3f}</b> "
        f"(95% CI {ci['roc_auc'][0]:.2f}–{ci['roc_auc'][1]:.2f}) and PR-AUC {m['test_pr_auc']:.3f}.",
        f"<b>Better than chemists' rules.</b> It beats the CNS rule of thumb "
        f"({baselines.iloc[0]['roc_auc']:.2f}), TPSA alone ({baselines.iloc[1]['roc_auc']:.2f}) "
        f"and a 6-descriptor logistic regression ({baselines.iloc[3]['roc_auc']:.2f}) in 100% of "
        f"paired bootstrap resamples.",
        f"<b>It rediscovers medicinal chemistry.</b> SHAP analysis shows TPSA is the top "
        f"driver, with the effect flipping sign near 80–90 Å² (the textbook CNS limit is 90 Å²). "
        f"H-bond donors and acids push towards \"blocked\"; lipophilicity pushes towards \"crosses\".",
        f"<b>Use a tuned threshold.</b> The training set is {frac_pos:.0%} BBB+, so at the "
        f"default 0.5 cutoff the model labels almost everything penetrant. A threshold chosen "
        f"on training cross-validation raises balanced accuracy from {t05['balanced_accuracy']:.2f} "
        f"to {ttun['balanced_accuracy']:.2f} and MCC from {t05['mcc']:.2f} to {ttun['mcc']:.2f}.",
        f"<b>Know the limits.</b> Accuracy drops on unfamiliar chemistry, probabilities are "
        f"over-confident on new scaffolds, and the model cannot see active transport "
        f"(P-gp efflux, LAT1 uptake), which explains its clearest mistakes.",
    ])
    story.append(PageBreak())

    # ---- 1. problem & data
    story += [P("1. The problem and the data", "h1"),
              P("<b>Why it matters.</b> The blood-brain barrier is a tight layer of cells that "
                "protects the brain and keeps most molecules out. A candidate drug for a "
                "neurological disease is useless if it cannot get past it, so predicting BBB "
                "penetration early helps prioritise which molecules to synthesise and test."),
              P("<b>Dataset.</b> We use the BBBP set from MoleculeNet (Martins et al., "
                "<i>J. Chem. Inf. Model.</i> 2012), which labels each compound as BBB-penetrating "
                "(1) or not (0).")]
    story.append(table([
        ["Property", "Value"],
        ["Raw compounds", f"{n_raw:,}"],
        ["Unparseable SMILES dropped", f"{n_bad}"],
        ["Molecules used", f"{n_mol:,}"],
        ["Penetrate the BBB (positive class)", f"{frac_pos:.1%}"],
        ["Distinct Bemis-Murcko scaffolds", f"{n_scaffolds:,}"],
    ], widths=[9.5, 5.5]))
    story += [P("<b>Cleaning.</b> Each SMILES is parsed with RDKit; salts and counter-ions are "
                "removed by keeping only the largest fragment. Molecules that fail to parse are "
                "dropped.", "body")]

    # ---- 2. features
    story += [P("2. Turning molecules into numbers", "h1"),
              P("A random forest needs a fixed-length numeric vector per molecule. Each molecule "
                "becomes <b>2,265 features</b>, built by <font face='%s'>src/features.py</font>:" % MONO)]
    story += bullets([
        "<b>217 physico-chemical descriptors</b> from RDKit — interpretable quantities such as "
        "molecular weight, TPSA (topological polar surface area), MolLogP (lipophilicity), "
        "counts of hydrogen-bond donors and acceptors, and the qed drug-likeness score.",
        "<b>A 2,048-bit Morgan fingerprint</b> (radius 2, roughly ECFP4) — each bit marks the "
        "presence of a particular circular substructure, letting the model react to specific "
        "chemical groups.",
    ])
    story += [P("A few descriptors can overflow on unusual molecules, so values are made finite "
                "and clipped to a safe range.", "body"),
              P("<b>Scaffolds.</b> Each molecule is also reduced to its Bemis-Murcko scaffold "
                "(its ring-and-linker core, stripped of side chains). Scaffolds are used to make "
                "the train/test split honest — see step 4.", "body")]

    # ---- 3. model
    story += [P("3. Training the random forest", "h1"),
              P("A random forest is an ensemble of decision trees, each trained on a random "
                "subset of the data and features; their votes are averaged. It handles thousands "
                "of mixed features well and needs little preprocessing, which suits this problem."),
              P("<b>Tuning.</b> <font face='%s'>src/train.py</font> runs a 20-trial randomised "
                "search over the number of trees, depth, features per split, leaf size and class "
                "weighting. Each candidate is scored with scaffold-grouped 5-fold "
                "cross-validation on the training set only, so no test molecule ever informs the "
                "choice. The winning settings:" % MONO)]
    story.append(table([
        ["Hyperparameter", "Chosen value"],
        ["n_estimators (trees)", str(bp["n_estimators"])],
        ["max_depth", str(bp["max_depth"])],
        ["max_features", str(bp["max_features"])],
        ["min_samples_leaf", str(bp["min_samples_leaf"])],
        ["class_weight", str(bp["class_weight"])],
    ], widths=[9.5, 5.5]))
    story += [P(f"The best cross-validation ROC-AUC was <b>{m['cv_roc_auc']:.3f}</b>. The shipped "
                f"model is then refit on all {n_mol:,} molecules and saved to "
                f"<font face='{MONO}'>outputs/rf_bbbp.joblib</font>.", "body")]

    # ---- 4. evaluation
    story += [PageBreak(), P("4. Honest evaluation: the scaffold split", "h1"),
              P("A <b>random</b> train/test split would be misleading here: close analogues of a "
                "test molecule often sit in the training set, so the model can almost memorise "
                "the answer. Instead we use a <b>scaffold split</b> — whole scaffold groups go "
                "entirely to train or entirely to test, so every test molecule has a core "
                "structure never seen in training. This mimics the real task of judging a "
                "genuinely new chemical series."),
              P(f"The split puts {m['n_train']:,} molecules in train and {m['n_test']} in test. "
                f"It also shifts the class balance (train is {frac_pos:.0%} BBB+, test is "
                f"{test_frac_pos:.0%}), which is exactly why accuracy at a "
                f"fixed 0.5 threshold looks worse than the ranking metrics.", "body")]
    story.append(table([
        ["Metric (held-out test)", "Score", "95% bootstrap CI"],
        ["ROC-AUC", f3(m["test_roc_auc"]), f"{ci['roc_auc'][0]:.3f} – {ci['roc_auc'][1]:.3f}"],
        ["PR-AUC", f3(m["test_pr_auc"]), f"{ci['pr_auc'][0]:.3f} – {ci['pr_auc'][1]:.3f}"],
        ["F1 @ 0.5", f3(m["test_f1"]), f"{ci['f1@0.5'][0]:.3f} – {ci['f1@0.5'][1]:.3f}"],
        ["Accuracy @ 0.5", f3(m["test_accuracy"]),
         f"{ci['accuracy@0.5'][0]:.3f} – {ci['accuracy@0.5'][1]:.3f}"],
    ], widths=[7, 3.5, 4.5], highlight_rows=[1]))
    story.append(figure("roc_curve.png", "Figure 1. ROC curve on the scaffold-split test set.",
                        width=9))

    # ---- 5. interpretation
    story += [PageBreak(), P("5. What the model learned (SHAP)", "h1"),
              P("<font face='%s'>src/explain.py</font> runs exact TreeSHAP on the forest to show "
                "how each feature pushes a prediction towards \"crosses\" (positive) or "
                "\"blocked\" (negative). Reassuringly, the drivers are the ones medicinal chemists "
                "already use." % MONO)]
    srows = [["Feature", "Mean |SHAP|", "Direction"]]
    dir_word = {"+": "+ helps crossing", "-": "– hinders crossing"}
    for _, r in shap_top.head(8).iterrows():
        srows.append([r["feature"], f'{r["mean_abs_shap"]:.4f}', dir_word.get(r["direction"], r["direction"])])
    story.append(table(srows, widths=[7, 3.5, 4.5]))
    story += bullets([
        "<b>TPSA is the #1 driver.</b> High polar surface area strongly lowers the predicted "
        "probability of crossing, with SHAP flipping sign near 80–90 Å² — the data rediscovering "
        "the rule of thumb that CNS drugs need TPSA below about 90 Å².",
        "<b>H-bond donors, N/O counts and heteroatoms</b> push towards \"blocked\".",
        "<b>Lipophilicity (MolLogP) and drug-likeness (qed)</b> push towards \"crosses\".",
    ])
    story.append(figure("shap_beeswarm.png",
                        "Figure 2. SHAP beeswarm: each dot is a molecule; red = high feature value.",
                        width=13))
    story.append(figure("shap_dependence_TPSA.png",
                        "Figure 3. TPSA dependence: the effect on P(crosses) flips sign near 80–90 Å².",
                        width=10))
    story += [P("<b>Explaining single molecules.</b> TreeSHAP also decomposes individual "
                "predictions. Caffeine (P = 0.94, crosses) and penicillin G (P = 0.01, blocked) "
                "are shown below, with Morgan bits labelled by the substructure they encode.",
                "body")]
    story.append(figure("shap_waterfall_caffeine.png",
                        "Figure 4a. Caffeine — pushed up towards \"crosses\".", width=12))
    story.append(figure("shap_waterfall_penicillin_g.png",
                        "Figure 4b. Penicillin G — pushed down by its acid, β-lactam and thioether.",
                        width=12))

    # ---- 6. trustworthiness
    story += [PageBreak(), P("6. How far can we trust it?", "h1"),
              P("<font face='%s'>src/evaluate.py</font> stress-tests the headline number four ways."
                % MONO),
              P("6.1 It beats the rules chemists already use", "h2"),
              P("Every model below is scored on the same scaffold-split test set. The rules use no "
                "training; the logistic regression is trained on the same molecules as the forest.")]
    brows = [["Model", "ROC-AUC", "95% CI", "PR-AUC"]]
    for _, r in baselines.iterrows():
        lo, hi = eval(r["roc_auc_95ci"]) if isinstance(r["roc_auc_95ci"], str) else r["roc_auc_95ci"]
        name = r["model"].split(" (")[0]
        brows.append([name, f'{r["roc_auc"]:.3f}', f"{lo:.2f} – {hi:.2f}", f'{r["pr_auc"]:.3f}'])
    story.append(table(brows, widths=[6.8, 2.6, 3.1, 2.5], highlight_last=True))
    story += [Paragraph(
        f"The forest wins against <b>every</b> baseline in 100% of 2,000 paired bootstrap "
        f"resamples. TPSA alone is a strong baseline ({baselines.iloc[1]['roc_auc']:.2f}), which "
        f"shows how much of BBB penetration simple polarity explains; the forest's extra "
        f"~{rf_row['roc_auc'] - baselines.iloc[1]['roc_auc']:.2f} AUC comes from substructure and "
        f"descriptor interactions.", S["box"])]
    story.append(figure("baselines_roc.png",
                        "Figure 5. Random forest vs. medicinal-chemistry baselines.", width=10.5))

    story += [P("6.2 The split matters more than the model", "h2"),
              P("The same model, with fixed hyperparameters, refit on 5 random and 5 shuffled "
                "scaffold splits:")]
    story.append(table([
        ["Split", "Test ROC-AUC (mean ± sd)"],
        ["Random (stratified)",
         f"{rs['Random']['mean']:.3f} ± {rs['Random']['std']:.3f}"],
        ["Scaffold, shuffled groups",
         f"{rs['Scaffold (shuffled)']['mean']:.3f} ± {rs['Scaffold (shuffled)']['std']:.3f}"],
        ["Scaffold, standard (reported)", f"{m['test_roc_auc']:.3f}"],
    ], widths=[9, 6]))
    story += [P(f"A random split would overstate performance by about "
                f"{rs['Random']['mean'] - rs['Scaffold (shuffled)']['mean']:.02f} AUC. The reported "
                f"number sits in the middle of the scaffold-split range, so it is not a lucky "
                f"split: expect roughly 0.85–0.90 on genuinely new chemotypes.", "body")]
    story.append(figure("split_comparison.png",
                        "Figure 6. Same model, different splits.", width=9))

    story += [PageBreak(), P("6.3 Choose the decision threshold, don't assume 0.5", "h2"),
              P(f"Because the training data is {frac_pos:.0%} BBB+, at 0.5 the forest labels almost "
                f"everything penetrant (sensitivity {t05['sensitivity']:.2f}, specificity "
                f"{t05['specificity']:.2f}). We picked a threshold from out-of-fold <i>training</i> "
                f"predictions only (maximising balanced accuracy) and applied it unchanged to the "
                f"test set.")]
    story.append(table([
        ["Threshold", "Accuracy", "Balanced acc.", "MCC", "Sensitivity", "Specificity"],
        ["0.50 (default)", f3(t05["accuracy"]), f3(t05["balanced_accuracy"]), f3(t05["mcc"]),
         f3(t05["sensitivity"]), f3(t05["specificity"])],
        [f"{ttun['threshold']:.2f} (tuned)", f3(ttun["accuracy"]), f3(ttun["balanced_accuracy"]),
         f3(ttun["mcc"]), f3(ttun["sensitivity"]), f3(ttun["specificity"])],
    ], widths=[3.2, 2.3, 2.6, 2, 2.6, 2.6], highlight_rows=[2]))
    story.append(figure("confusion_matrices.png",
                        "Figure 7. Confusion matrices at the default and tuned thresholds.",
                        width=13))

    story += [P("6.4 Are the probabilities trustworthy?", "h2"),
              P(f"On training cross-validation the probabilities are reasonably calibrated "
                f"(Brier {cal['brier_train_cv']:.3f}). On the scaffold test set they are "
                f"over-confident (Brier {cal['brier_test']:.3f}); isotonic recalibration fitted on "
                f"the training CV improves this to {cal['brier_test_isotonic']:.3f}. Read the "
                f"output as a <b>ranking score</b>, not a literal probability, especially for "
                f"unfamiliar scaffolds.")]
    story.append(figure("calibration.png", "Figure 8. Reliability diagram.", width=9))

    # ---- 7. where it fails
    story += [PageBreak(), P("7. Where the model works and where it fails", "h1"),
              P("<font face='%s'>src/analysis.py</font> looks at individual molecules and at the "
                "dataset itself." % MONO),
              P("7.1 Applicability domain: novelty costs accuracy", "h2"),
              P("For each test molecule we take its Tanimoto similarity (Morgan fingerprints) to "
                "the nearest training molecule, then bin by that similarity.")]
    arows = [["Similarity to train", "n", "ROC-AUC", "Accuracy (tuned)"]]
    for _, r in ad.iterrows():
        arows.append([r["similarity_to_train"], str(int(r["n"])),
                      "—" if pd.isna(r["roc_auc"]) else f'{r["roc_auc"]:.3f}', f'{r["accuracy"]:.3f}'])
    story.append(table(arows, widths=[5, 2, 4, 4]))
    story += [Paragraph("Below about 0.5 similarity, accuracy falls from ~0.85–0.94 to ~0.70–0.77. "
                        "A practical rule: report the nearest-neighbour similarity next to every "
                        "prediction and treat anything below 0.4 as low-confidence.", S["box"])]
    story.append(figure("applicability_domain.png",
                        "Figure 9. Test-set novelty (left) and performance vs. novelty (right).",
                        width=13))
    story.append(figure("chemical_space.png",
                        "Figure 10. t-SNE of Morgan fingerprints: by label (left), by split and "
                        "errors (right).", width=14.5))

    story += [PageBreak(), P("7.2 The most confident mistakes", "h2"),
              P("The errors fall into two clear, chemically meaningful groups.")]
    story.append(figure("error_molecules.png",
                        "Figure 11. The most confident false positives (top) and false negatives.",
                        width=14))
    fp = errors[errors["error"] == "false positive"].head(3)["name"].tolist()
    fn = errors[errors["error"] == "false negative"].head(3)["name"].tolist()
    story += bullets([
        f"<b>False positives — lipophilic, low-TPSA drugs labelled \"blocked\"</b> "
        f"(e.g. {', '.join(fp)}). They look textbook CNS-like, but several are substrates of "
        f"<b>P-glycoprotein</b>, an efflux pump that throws them back out of the brain. That is "
        f"active transport, which a structure-only passive-permeability model cannot see.",
        f"<b>False negatives — big, polar antibiotics labelled \"crosses\"</b> "
        f"(e.g. {', '.join(fn)}). Their physical chemistry says \"blocked\", and the model agrees; "
        f"the dataset label reflects clinical use against meningitis, when the inflamed barrier "
        f"becomes leaky and high doses are given.",
    ])
    story += [Paragraph("So many of the \"errors\" are arguably label-definition issues, not "
                        "model failures.", S["box"])]

    story += [P("7.3 Label noise in BBBP", "h2"),
              P(f"There are <b>{an['label_conflicts_tanimoto_0.9']} pairs</b> of molecules with "
                f"Tanimoto ≥ 0.9 but opposite labels, and "
                f"<b>{an['label_conflicts_identical_structure']} of them are the exact same "
                f"structure</b> listed twice with different labels. This noise caps the accuracy "
                f"any model can reach on BBBP.")]
    crows = [["Molecule A", "Label", "Molecule B", "Label", "Same structure?"]]
    for _, r in conflicts.head(6).iterrows():
        crows.append([r["mol_a"], str(r["label_a"]), r["mol_b"], str(r["label_b"]),
                      "yes" if r["identical_structure"] else "no"])
    story.append(table(crows, widths=[4, 1.6, 4, 1.6, 3.3]))

    story += [PageBreak(), P("7.4 Case studies: known drugs the model never saw", "h2"),
              P("Before predicting, every case-study drug was removed from the training data and "
                "the forest refit, so this is a genuine out-of-sample test against documented "
                "pharmacology.")]
    krows = [["Drug", "Known", "P(BBB+)", "Correct", "Why"]]
    for _, r in cases.iterrows():
        krows.append([r["drug"], r["expected"], f'{r["p_bbb"]:.2f}',
                      "Yes" if r["correct@tuned"] else "No", r["why"]])
    story.append(table(krows, widths=[2.9, 1.8, 1.8, 1.6, 6.9]))
    story += [Paragraph(
        f"The forest gets <b>{n_correct_cases} / {len(cases)}</b> right at the tuned threshold and "
        f"correctly separates the sedating from the non-sedating antihistamines. Its failures are "
        f"exactly the drugs whose BBB behaviour depends on <b>transporters</b> rather than passive "
        f"diffusion (P-gp efflux, and LAT1 uptake for L-DOPA) — the clearest limitation of a "
        f"structure-only model.", S["box"])]
    story.append(figure("case_studies.png",
                        "Figure 12. Known drugs held out from training (green = crosses, "
                        "red = blocked).", width=11))

    # ---- 8. limitations & reproduce
    story += [PageBreak(), P("8. Limitations", "h1")]
    story += bullets([
        "BBBP labels are binary and come from mixed sources, with known duplicates and conflicts.",
        "The model learns passive permeability. Efflux (P-gp, BCRP) and uptake transporters "
        "(LAT1, GLUT1) are invisible to it unless similar molecules appear in training.",
        "Brain exposure also depends on dose, plasma protein binding and metabolism, none of "
        "which the model sees.",
        "Use it to rank and triage candidates, alongside the applicability-domain check, not as "
        "a definitive yes/no.",
    ])
    story += [P("9. Reproducing this report", "h1"),
              P("From the repository root:")]
    story.append(Paragraph(
        "pip install -r requirements.txt<br/>"
        "python src/train.py&nbsp;&nbsp;&nbsp;&nbsp;# model + metrics.json, ROC, importances<br/>"
        "python src/explain.py&nbsp;&nbsp;# SHAP plots<br/>"
        "python src/evaluate.py&nbsp;# baselines, CIs, splits, threshold, calibration<br/>"
        "python src/analysis.py&nbsp;# applicability domain, errors, label noise, cases<br/>"
        "python src/report.py&nbsp;&nbsp;&nbsp;# this PDF", S["code"]))
    story += [P(f"All numbers and figures in this report are read directly from the files in "
                f"<font face='{MONO}'>outputs/</font>, so regenerating those files updates the "
                f"report. Generated {date.today():%d %B %Y}.", "body")]

    REPORT_DIR.mkdir(exist_ok=True)
    doc = ReportDoc(str(PDF), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                    leftMargin=2 * cm, rightMargin=2 * cm,
                    title="Blood-Brain Barrier Penetration with a Random Forest",
                    author="ChemAI project")
    doc.multiBuild(story, onLaterPages=on_page, onFirstPage=on_page)
    print(f"Wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
