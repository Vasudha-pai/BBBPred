#  Blood-Brain Barrier Penetration with a Random Forest

A drug for a brain disease only works if it can actually **reach the brain**. Most
molecules are blocked by the blood-brain barrier (BBB). This project predicts, from a
molecule's structure alone (its SMILES string), whether it will cross the BBB — using a
**single random forest** model.

> 📄 **[Full step-by-step report (PDF)](report/BBBP_Report.pdf)** — data, model, SHAP, trustworthiness and error analysis in one document, regenerated from `outputs/` by `src/report.py`.

## Results

Evaluated on a **scaffold split**: test molecules have core structures (Bemis-Murcko
scaffolds) never seen in training, which mimics predicting genuinely new drug candidates
and is much harder than a random split.

| Metric (held-out test, 408 molecules) | Score | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | **0.882** | 0.847 – 0.914 |
| PR-AUC | 0.897 | 0.857 – 0.932 |
| Balanced accuracy (tuned threshold 0.85) | 0.789 | |
| MCC (tuned threshold 0.85) | 0.577 | |
| F1 / accuracy at threshold 0.5 | 0.791 / 0.726 | 0.75 – 0.83 / 0.68 – 0.77 |
| 5-fold scaffold-grouped CV ROC-AUC (train) | 0.890 | |

The scaffold split shifts class balance (82% BBB+ in train vs 54% in test), which is why
accuracy at the default 0.5 threshold trails the ranking metrics. A threshold chosen on
training-set cross-validation fixes most of that (see [How far can we trust it?](#how-far-can-we-trust-it)).

![ROC curve](outputs/roc_curve.png)

### What the model learned

The most important features are the ones medicinal chemists already use to judge brain
penetration — a good sanity check:

- **TPSA** (topological polar surface area) — polar molecules struggle to cross
- **NumHDonors / NHOHCount** — hydrogen-bond donors hurt permeability
- **MolLogP** — lipophilicity helps passive diffusion through membranes
- **qed** — overall drug-likeness

![Feature importance](outputs/feature_importance.png)

## SHAP analysis: why the model decides what it decides

`src/explain.py` runs exact TreeSHAP on the final random forest (400 sampled molecules)
to show how each feature pushes a prediction towards **crosses the BBB** (positive SHAP)
or **blocked** (negative SHAP).

![SHAP beeswarm](outputs/shap_beeswarm.png)

Key findings (red = high feature value):

- **TPSA is the #1 driver.** High polar surface area strongly lowers P(BBB+). The
  dependence plot shows SHAP flipping sign at **~80-90 Å²**, rediscovering the
  medicinal-chemistry rule of thumb that CNS drugs need TPSA < 90 Å² — learned from data alone.
- **H-bond donors, N/O counts and heteroatoms** all push towards "blocked".
- **Lipophilicity (MolLogP) and drug-likeness (qed)** push towards "crosses".
- **Carboxylic acids (`fr_COO2`)** are penalised: they are charged at blood pH and
  rarely cross membranes.
- The "sum of other features" row is large because signal is spread across thousands of
  correlated fingerprint bits and descriptors — individual features each contribute a
  little, but together they separate the classes clearly.

![TPSA dependence](outputs/shap_dependence_TPSA.png)

### Explaining single molecules

Morgan fingerprint bits are labelled with the substructure they encode.

| Caffeine — P(BBB+) = 0.94 | Penicillin G — P(BBB+) = 0.01 |
|---|---|
| ![Caffeine](outputs/shap_waterfall_caffeine.png) | ![Penicillin G](outputs/shap_waterfall_penicillin_g.png) |

Penicillin G (which indeed does not reach the brain well) is pushed down by its
carboxylic acid, β-lactam and thioether groups.

## How far can we trust it?

`src/evaluate.py` stress-tests the headline number.

### 1. It beats the rules chemists already use

Every model below is scored on the same scaffold-split test set. Rules use no training;
the logistic regression is trained on the same training molecules as the forest.

| Model | ROC-AUC (95% CI) | PR-AUC |
|---|---|---|
| CNS rule of thumb (TPSA < 90, MW < 450, HBD ≤ 3) | 0.747 (0.70 – 0.79) | 0.717 |
| CNS MPO score (Wager 2010, 4 of 6 components) | 0.735 (0.68 – 0.78) | 0.746 |
| TPSA alone | 0.826 (0.78 – 0.87) | 0.843 |
| Logistic regression on 6 classic descriptors | 0.810 (0.77 – 0.85) | 0.807 |
| **Random forest (this project)** | **0.882 (0.85 – 0.91)** | **0.897** |

The forest wins against every baseline in **100% of 2,000 paired bootstrap resamples**.
TPSA alone is a strong baseline (0.83), which shows how much of BBB penetration polarity
explains. The forest's extra ~0.06 AUC comes from substructure and descriptor
interactions. Our CNS MPO leaves out the logD and pKa terms (RDKit can't compute them), so
it understates the published score.

![Baselines](outputs/baselines_roc.png)

### 2. The split matters more than the model

The same model with fixed hyperparameters, refit on 5 random and 5 shuffled scaffold splits:

| Split | Test ROC-AUC (mean ± sd) |
|---|---|
| Random (stratified) | 0.933 ± 0.008 |
| Scaffold, shuffled groups | 0.885 ± 0.020 |
| Scaffold, standard (reported above) | 0.882 |

A random split would overstate performance by ~0.05 AUC, because near-copies of test
molecules sit in the training set. The reported number sits in the middle of the
scaffold-split range, so it isn't a lucky split. Expect **0.85 – 0.90** on genuinely new
chemotypes.

![Split comparison](outputs/split_comparison.png)

### 3. Choose the decision threshold, don't assume 0.5

The training data is 82% BBB+, so the forest is biased towards "crosses". At 0.5 it labels
almost everything as penetrant (sensitivity 0.96, specificity 0.45). We picked a threshold
from **out-of-fold training predictions only** (maximising Youden's J, i.e. balanced
accuracy) and applied it unchanged to the test set:

| Threshold | Accuracy | Balanced acc. | MCC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| 0.50 (default) | 0.725 | 0.703 | 0.481 | 0.955 | 0.452 |
| **0.85 (tuned on train CV)** | **0.784** | **0.789** | **0.577** | 0.734 | 0.844 |

![Confusion matrices](outputs/confusion_matrices.png)

### 4. Probabilities are over-confident on new chemotypes

On training CV the probabilities are reasonably calibrated (Brier 0.089). On the scaffold
test set they're over-confident in the 0.5 – 0.9 range (Brier 0.182). Isotonic
recalibration fitted on the training CV improves this to 0.160, but some gap remains
because the class balance shifts between train and test. **Read P(BBB+) as a ranking
score, not a literal probability**, especially for unfamiliar scaffolds.

![Calibration](outputs/calibration.png)

## Where the model works and where it fails

`src/analysis.py` looks at individual molecules and at the dataset itself.

### Applicability domain: novelty costs accuracy

For each test molecule we take its Tanimoto similarity (Morgan fingerprints) to the
*nearest* training molecule:

| Similarity to train | n | ROC-AUC | Accuracy (tuned) |
|---|---|---|---|
| < 0.3 | 105 | 0.838 | 0.752 |
| 0.3 – 0.4 | 86 | 0.792 | 0.698 |
| 0.4 – 0.5 | 68 | 0.850 | 0.765 |
| 0.5 – 0.7 | 118 | 0.918 | 0.847 |
| ≥ 0.7 | 31 | 0.969 | 0.935 |

Below ~0.5 similarity, accuracy falls from ~0.85–0.94 to ~0.70–0.77. Wrong predictions
average 0.39 similarity vs 0.46 for correct ones. **A practical rule:** report the
nearest-neighbour similarity next to every prediction, and treat predictions below 0.4 as
low-confidence.

![Applicability domain](outputs/applicability_domain.png)

### Chemical space

A t-SNE map of the Morgan fingerprints. Blocked compounds cluster into a few islands
(β-lactam antibiotics, tetracyclines, steroids/glycosides). Brain-penetrant molecules fill
the large central drug-like region. The scaffold split scatters test molecules across the
map, and errors (red ×) concentrate where the two classes mix.

![Chemical space](outputs/chemical_space.png)

### The most confident mistakes

![Error molecules](outputs/error_molecules.png)

The errors fall into two clear groups (full table in `outputs/error_analysis.csv`):

- **False positives: lipophilic, low-TPSA drugs labelled "blocked".** Examples include
  domperidone, astemizole, bepridil and phenoxybenzamine. They look textbook CNS-like
  (TPSA 12 – 80 Å², logP 3 – 5). Several are known substrates of **P-glycoprotein**, an
  efflux pump that throws them back out of the brain. That is active transport, which a
  structure-only passive-permeability model cannot see.
- **False negatives: big, polar antibiotics labelled "crosses".** Examples include
  ceftriaxone, ceftazidime, cefoperazone, piperacillin, minocycline and rolitetracycline
  (TPSA 156 – 220 Å²). They are used against meningitis because they enter the CNS when
  the barrier is **inflamed**, and at high doses. The model's "blocked" call matches their
  physical chemistry. The dataset label reflects clinical use.

So many of the "errors" are arguably label-definition issues, not model failures.

### Label noise in BBBP

There are **15 pairs** of molecules with Tanimoto ≥ 0.9 but opposite labels, and **11 of
them are the exact same structure** listed twice. Examples are propranolol (1 vs 0),
aspirin, atropine, indomethacin, trimetrexate and loperamide/BRL53080. Others are
stereoisomers the fingerprint can't distinguish (quinine/quinidine,
hyoscyamine/atropine). This noise caps the accuracy any model can reach on BBBP. See
`outputs/label_conflicts.csv`.

### Case studies: known drugs the model has never seen

Before predicting, we removed every case-study drug from the training data and refit the
forest, so this is a real out-of-sample test:

| Drug | Known behaviour | P(BBB+) | Correct (tuned thr.) | Why |
|---|---|---|---|---|
| Diazepam | crosses | 1.00 | ✓ | lipophilic, passive diffusion |
| Fluoxetine | crosses | 0.94 | ✓ | passive diffusion |
| Diphenhydramine | crosses | 0.98 | ✓ | sedating 1st-gen antihistamine |
| Hydroxyzine | crosses | 0.98 | ✓ | sedating 1st-gen antihistamine |
| Fexofenadine | blocked | 0.47 | ✓ | zwitterion + P-gp efflux (non-drowsy) |
| Cetirizine | blocked | 0.83 | ✓ | hydroxyzine + a carboxylic acid |
| Dopamine | blocked | 0.48 | ✓ | polar neurotransmitter |
| Atenolol | blocked | 0.79 | ✓ | hydrophilic beta-blocker |
| Gabapentin | crosses | 0.85 | ✓ | amino acid, LAT1 transporter |
| Loperamide | blocked | 0.88 | ✗ | P-gp efflux keeps this opioid out |
| Domperidone | blocked | 0.94 | ✗ | P-gp efflux |
| L-DOPA | crosses | 0.37 | ✗ | polar, but carried in by LAT1 |

The forest gets **9/12** right at the tuned threshold (7/12 at 0.5). It correctly tells
apart the sedating and non-sedating antihistamine pairs. Its three failures are exactly the
drugs whose BBB behaviour depends on **transporters**, not passive diffusion: two P-gp
efflux substrates and L-DOPA, which LAT1 carries into the brain. This is the clearest
limitation of a structure-only model.

![Case studies](outputs/case_studies.png)

### Limitations

- BBBP labels are binary and come from mixed sources, with known duplicates and conflicts.
- The model learns passive permeability. Efflux (P-gp, BCRP) and uptake transporters
  (LAT1, GLUT1) are invisible to it unless similar molecules are in the training data.
- Brain exposure also depends on dose, plasma protein binding and metabolism, none of
  which the model sees.
- Use it to rank and triage candidates, alongside the applicability-domain check, not as
  a definitive yes/no.

## Approach

1. **Data** — [BBBP](https://moleculenet.org/datasets-1) from MoleculeNet: 2,050 compounds
   labelled BBB-penetrating (1) or not (0) (Martins et al., *J. Chem. Inf. Model.* 2012, 52, 1686). 11 unparseable SMILES are dropped; salts and
   counter-ions are stripped (largest fragment kept).
2. **Features** (`src/features.py`) — 217 RDKit physico-chemical descriptors + 2048-bit
   Morgan fingerprint (radius 2, ≈ ECFP4).
3. **Model** (`src/train.py`) — `RandomForestClassifier`, tuned with a 20-trial randomized
   search over trees, depth, `max_features`, leaf size and class weighting, using
   scaffold-grouped 5-fold CV on the training set only. Best:
   800 trees, `max_features="log2"`, `max_depth=40`, `class_weight="balanced_subsample"`.
4. **Final model** — refit on all data and saved to `outputs/rf_bbbp.joblib`.

## Usage

```bash
pip install -r requirements.txt
python src/train.py          # ~5 min on a laptop; writes outputs/
python src/explain.py        # ~3 min; SHAP plots in outputs/
python src/evaluate.py       # ~2 min; baselines, CIs, split comparison, threshold, calibration
python src/analysis.py       # ~1 min; applicability domain, errors, label noise, case studies
python src/report.py         # build report/BBBP_Report.pdf from everything above
python src/predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"   # caffeine
```

## Project structure

```
data/BBBP.csv            MoleculeNet BBBP dataset
src/features.py          SMILES → descriptors + fingerprints, scaffolds
src/data.py              data loading, scaffold split, shared train-split model
src/train.py             RF tuning, evaluation, plots
src/predict.py           predict BBB penetration for new SMILES
src/explain.py           SHAP analysis: global + per-molecule explanations
src/evaluate.py          trustworthiness: baselines, bootstrap CIs, splits, threshold, calibration
src/analysis.py          applicability domain, error analysis, label noise, case studies, t-SNE
src/report.py            assemble the full step-by-step PDF report from outputs/
outputs/                 metrics, plots and CSV/JSON reports from all scripts
report/BBBP_Report.pdf   generated step-by-step report (data to error analysis)
```
