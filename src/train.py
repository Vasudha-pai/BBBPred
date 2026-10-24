"""Train a random forest to predict blood-brain barrier penetration (BBBP).

Usage: python src/train.py
"""
import json
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from features import feature_names, featurize_many, scaffold, smiles_to_mol

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "BBBP.csv"
OUT = ROOT / "outputs"
SEED = 42


def scaffold_split(scaffolds, test_frac=0.2):
    """Put whole scaffold groups in train or test so the test set has unseen chemotypes.

    Largest groups go to train first (the standard MoleculeNet recipe), leaving
    rarer scaffolds for test - a harder, more realistic estimate than a random split.
    """
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n_train = int((1 - test_frac) * len(scaffolds))
    train, test = [], []
    for g in ordered:
        (train if len(train) + len(g) <= n_train else test).extend(g)
    return np.array(train), np.array(test)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    df["mol"] = df["smiles"].map(smiles_to_mol)
    n_bad = int(df["mol"].isna().sum())
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    print(f"Loaded {len(df)} molecules ({n_bad} unparseable SMILES dropped)")
    print(f"Class balance: {df['p_np'].mean():.1%} penetrate the BBB")

    X = featurize_many(df["mol"])
    y = df["p_np"].to_numpy()
    scaffolds = df["mol"].map(scaffold).to_numpy()
    train_idx, test_idx = scaffold_split(scaffolds)
    print(f"Scaffold split: {len(train_idx)} train / {len(test_idx)} test")

    # Tune on train only, with scaffold-grouped CV folds.
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        param_distributions={
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 20, 40],
            "max_features": ["sqrt", "log2", 0.1],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        n_iter=20,
        scoring="roc_auc",
        cv=GroupKFold(n_splits=5),
        random_state=SEED,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X[train_idx], y[train_idx], groups=scaffolds[train_idx])
    model = search.best_estimator_
    print(f"Best CV ROC-AUC: {search.best_score_:.3f} with {search.best_params_}")

    proba = model.predict_proba(X[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = y[test_idx]
    metrics = {
        "cv_roc_auc": round(search.best_score_, 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "best_params": search.best_params_,
    }
    print(json.dumps(metrics, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Random forest (AUC = {metrics['test_roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("BBBP - scaffold-split test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close()

    # Top feature importances
    imp = pd.Series(model.feature_importances_, index=feature_names()).nlargest(20)[::-1]
    imp.to_frame("importance").to_csv(OUT / "top_features.csv")
    plt.figure(figsize=(6, 6))
    plt.barh(imp.index, imp.values)
    plt.xlabel("Mean decrease in impurity")
    plt.title("Top 20 features")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    # Refit on all data for the shipped model.
    final = RandomForestClassifier(random_state=SEED, n_jobs=-1, **search.best_params_).fit(X, y)
    joblib.dump(final, OUT / "rf_bbbp.joblib", compress=3)
    print(f"Saved model and reports to {OUT}")


if __name__ == "__main__":
    main()
