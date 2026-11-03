"""How much can we trust the random forest? Run after train.py.

1. Baselines   - is the RF better than the rules medicinal chemists already use?
2. Uncertainty - bootstrap 95% confidence intervals on the test metrics.
3. Splits      - repeated random vs scaffold splits: how much does the split matter?
4. Threshold   - pick a decision threshold from training CV instead of assuming 0.5.
5. Calibration - when the model says 80%, is it right 80% of the time?

Usage: python src/evaluate.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import OUT, SEED, load_split, make_rf, scaffold_split
from features import DESCRIPTOR_NAMES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_BOOT = 2000
N_REPEATS = 5
CLASSIC = ["TPSA", "MolLogP", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]


def desc(X, name):
    return X[:, DESCRIPTOR_NAMES.index(name)]


def ramp(x, lo, hi):
    """1 at/below lo, 0 at/above hi, linear in between (an MPO 'desirability' function)."""
    return np.clip((hi - x) / (hi - lo), 0, 1)


def cns_mpo(X):
    """Simplified CNS MPO score (Wager et al. 2010), 0-4.

    Uses the four components RDKit can compute (cLogP, MW, TPSA, HBD); the original
    also scores logD and pKa, which need dedicated predictors.
    """
    tpsa = desc(X, "TPSA")
    tpsa_score = np.where(tpsa < 40, np.clip((tpsa - 20) / 20, 0, 1), ramp(tpsa, 90, 120))
    return (ramp(desc(X, "MolLogP"), 3, 5) + ramp(desc(X, "MolWt"), 360, 500)
            + tpsa_score + ramp(desc(X, "NumHDonors"), 0.5, 3.5))


def cns_rule(X):
    """Rule of thumb for CNS drugs: TPSA < 90, MW < 450, HBD <= 3 (1 = predicted to cross)."""
    return ((desc(X, "TPSA") < 90) & (desc(X, "MolWt") < 450)
            & (desc(X, "NumHDonors") <= 3)).astype(float)


def bootstrap(y, scores, metric, n=N_BOOT, seed=SEED):
    """Percentile bootstrap over test molecules; `scores` may be a list (paired resampling)."""
    rng = np.random.default_rng(seed)
    scores = scores if isinstance(scores, list) else [scores]
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        out.append([metric(y[idx], s[idx]) for s in scores])
    return np.array(out)


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def threshold_metrics(y, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": round(float(t), 3),
        "accuracy": round(accuracy_score(y, pred), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 3),
        "f1": round(f1_score(y, pred), 3),
        "mcc": round(matthews_corrcoef(y, pred), 3),
        "sensitivity": round(tp / (tp + fn), 3),
        "specificity": round(tn / (tn + fp), 3),
    }


def baselines(X, y, tr, te, rf_proba):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cols = [DESCRIPTOR_NAMES.index(c) for c in CLASSIC]
    lr.fit(X[tr][:, cols], y[tr])
    scores = {
        "CNS rule (TPSA<90, MW<450, HBD<=3)": cns_rule(X[te]),
        "TPSA alone (lower = crosses)": -desc(X[te], "TPSA"),
        "CNS MPO score (4-component)": cns_mpo(X[te]),
        "Logistic regression (6 descriptors)": lr.predict_proba(X[te][:, cols])[:, 1],
        "Random forest (this project)": rf_proba,
    }
    yt = y[te]
    boot = bootstrap(yt, list(scores.values()), roc_auc_score)
    rows = []
    for j, (name, s) in enumerate(scores.items()):
        rows.append({
            "model": name,
            "roc_auc": round(roc_auc_score(yt, s), 3),
            "roc_auc_95ci": ci(boot[:, j]),
            "pr_auc": round(average_precision_score(yt, s), 3),
            # Paired bootstrap: how often does the RF beat this model on a resampled test set?
            "rf_better_in_%_of_bootstraps": round(100 * float((boot[:, -1] > boot[:, j]).mean()), 1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "baselines.csv", index=False)

    plt.figure(figsize=(5.5, 5.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(yt, s)
        style = "-" if "Random forest" in name else "--"
        plt.plot(fpr, tpr, style, lw=2 if "Random" in name else 1.2,
                 label=f"{name.split(' (')[0]} ({roc_auc_score(yt, s):.2f})")
    plt.plot([0, 1], [0, 1], ":", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Random forest vs. medicinal-chemistry baselines")
    plt.legend(loc="lower right", fontsize=8, title="ROC-AUC", title_fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "baselines_roc.png", dpi=150)
    plt.close()
    return table


def repeated_splits(df, X, y):
    """Refit the RF (fixed hyperparameters) on several random and scaffold splits."""
    scaffolds = df["scaffold"].to_numpy()
    rows = []
    for seed in range(N_REPEATS):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        splits = {"Random": (tr, te), "Scaffold (shuffled)": scaffold_split(scaffolds, seed=seed)}
        for kind, (tr, te) in splits.items():
            p = make_rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            rows.append({"split": kind, "seed": seed, "roc_auc": roc_auc_score(y[te], p),
                         "test_frac_bbb+": y[te].mean()})
            print(f"  {kind:20s} seed {seed}: ROC-AUC {rows[-1]['roc_auc']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "repeated_splits.csv", index=False)
    return res


def plot_splits(res, default_auc):
    summary = res.groupby("split")["roc_auc"].agg(["mean", "std"])
    order = ["Random", "Scaffold (shuffled)"]
    plt.figure(figsize=(5, 4))
    for i, kind in enumerate(order):
        vals = res.loc[res["split"] == kind, "roc_auc"]
        plt.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, zorder=3)
        plt.errorbar(i + 0.22, summary.loc[kind, "mean"], yerr=summary.loc[kind, "std"],
                     fmt="o", color="black", capsize=4)
    plt.scatter([2], [default_auc], marker="*", s=150, color="crimson", zorder=3)
    plt.xticks([0, 1, 2], ["Random\n(5 seeds)", "Scaffold, shuffled\n(5 seeds)",
                           "Scaffold, standard\n(reported)"])
    plt.ylabel("Test ROC-AUC")
    plt.title("Same model, different splits")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "split_comparison.png", dpi=150)
    plt.close()
    return summary


def main():
    df, X, y, tr, te, model = load_split()
    yt = y[te]
    proba = model.predict_proba(X[te])[:, 1]
    report = {}

    print("1. Baselines")
    table = baselines(X, y, tr, te, proba)
    print(table.to_string(index=False))
    report["baselines"] = table.to_dict(orient="records")

    print("2. Bootstrap confidence intervals")
    t05 = (proba >= 0.5).astype(int)
    report["test_metrics_95ci"] = {
        "roc_auc": ci(bootstrap(yt, proba, roc_auc_score)[:, 0]),
        "pr_auc": ci(bootstrap(yt, proba, average_precision_score)[:, 0]),
        "f1@0.5": ci(bootstrap(yt, t05, f1_score)[:, 0]),
        "accuracy@0.5": ci(bootstrap(yt, t05, accuracy_score)[:, 0]),
    }
    print(report["test_metrics_95ci"])

    print("3. Repeated random vs scaffold splits")
    res = repeated_splits(df, X, y)
    summary = plot_splits(res, roc_auc_score(yt, proba))
    report["repeated_splits_roc_auc"] = {
        k: {"mean": round(r["mean"], 3), "std": round(r["std"], 3)} for k, r in summary.iterrows()}
    print(summary)

    print("4. Threshold from out-of-fold training predictions")
    oof = cross_val_predict(make_rf(), X[tr], y[tr], groups=df["scaffold"].to_numpy()[tr],
                            cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(y[tr], oof)
    t_best = float(thr[np.argmax(tpr - fpr)])  # Youden's J = max balanced accuracy
    report["threshold"] = {"default_0.5": threshold_metrics(yt, proba, 0.5),
                           "tuned_on_train_cv": threshold_metrics(yt, proba, t_best)}
    print(json.dumps(report["threshold"], indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (label, t) in zip(axes, [("Threshold 0.50", 0.5), (f"Tuned threshold {t_best:.2f}", t_best)]):
        cm = confusion_matrix(yt, (proba >= t).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Blocked", "Crosses"])
        ax.set_yticks([0, 1], ["Blocked", "Crosses"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        m = threshold_metrics(yt, proba, t)
        ax.set_title(f"{label}\nbal. acc {m['balanced_accuracy']:.2f}, MCC {m['mcc']:.2f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close()

    print("5. Calibration")
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y[tr])
    proba_iso = iso.predict(proba)
    report["calibration"] = {
        "brier_train_cv": round(brier_score_loss(y[tr], oof), 4),
        "brier_test": round(brier_score_loss(yt, proba), 4),
        "brier_test_isotonic": round(brier_score_loss(yt, proba_iso), 4),
        "brier_test_always_predict_train_rate": round(brier_score_loss(yt, np.full(len(yt), y[tr].mean())), 4),
    }
    print(report["calibration"])
    plt.figure(figsize=(5, 5))
    for label, yy, pp in [("Train (out-of-fold CV)", y[tr], oof), ("Test (scaffold split)", yt, proba),
                          ("Test, isotonic-calibrated", yt, proba_iso)]:
        frac, mean = calibration_curve(yy, pp, n_bins=10, strategy="quantile")
        plt.plot(mean, frac, "o-", label=f"{label} (Brier {brier_score_loss(yy, pp):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted P(crosses BBB)")
    plt.ylabel("Observed fraction that cross")
    plt.title("Reliability diagram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=150)
    plt.close()

    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved evaluation reports to {OUT}")


if __name__ == "__main__":
    main()
