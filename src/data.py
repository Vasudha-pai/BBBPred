"""Shared data loading, scaffold split and the train-split model used by the reports."""
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import featurize_many, scaffold, smiles_to_mol

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "BBBP.csv"
OUT = ROOT / "outputs"
SEED = 42


def load_dataset():
    """BBBP molecules with salts stripped; unparseable SMILES dropped."""
    df = pd.read_csv(DATA)
    df["mol"] = df["smiles"].map(smiles_to_mol)
    n_bad = int(df["mol"].isna().sum())
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    df["scaffold"] = df["mol"].map(scaffold)
    return df, n_bad


def scaffold_split(scaffolds, test_frac=0.2, seed=None):
    """Put whole scaffold groups in train or test so the test set has unseen chemotypes.

    With seed=None the largest groups go to train first (the standard MoleculeNet
    recipe), leaving rarer scaffolds for test - a harder, more realistic estimate than
    a random split. With a seed, groups are shuffled instead, giving a different (but
    still scaffold-disjoint) split for repeated evaluation.
    """
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    if seed is None:
        ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    else:
        ordered = list(groups.values())
        np.random.default_rng(seed).shuffle(ordered)
    n_train = int((1 - test_frac) * len(scaffolds))
    train, test = [], []
    for g in ordered:
        (train if len(train) + len(g) <= n_train else test).extend(g)
    return np.array(train), np.array(test)


def best_params():
    """Hyperparameters chosen by train.py's scaffold-grouped search."""
    return json.loads((OUT / "metrics.json").read_text())["best_params"]


def make_rf(**overrides):
    return RandomForestClassifier(random_state=SEED, n_jobs=-1, **{**best_params(), **overrides})


def load_split():
    """Dataset, features and the default scaffold split, plus an RF fitted on train only.

    The fitted model is cached in outputs/rf_train_split.joblib (git-ignored) so the
    report scripts don't refit it each time. Delete it after re-running train.py.
    """
    df, _ = load_dataset()
    X = featurize_many(df["mol"])
    y = df["p_np"].to_numpy()
    train_idx, test_idx = scaffold_split(df["scaffold"].to_numpy())
    cache = OUT / "rf_train_split.joblib"
    if cache.exists():
        model = joblib.load(cache)
    else:
        model = make_rf().fit(X[train_idx], y[train_idx])
        joblib.dump(model, cache, compress=3)
    return df, X, y, train_idx, test_idx, model
