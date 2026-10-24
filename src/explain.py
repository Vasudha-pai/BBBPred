"""SHAP analysis of the trained BBBP random forest.

Explains which molecular properties and substructures push each prediction
towards "crosses the blood-brain barrier" (positive SHAP) or "blocked"
(negative SHAP). Run after train.py.

Usage: python src/explain.py
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from features import FP_BITS, feature_names, featurize_many, smiles_to_mol

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "BBBP.csv"
OUT = ROOT / "outputs"
MODEL = OUT / "rf_bbbp.joblib"
SEED = 42
N_SAMPLE = 400  # exact TreeSHAP on 800 deep trees is slow; a sample is plenty for global plots
EXAMPLES = {
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
}


def morgan_bit_fragment(mols, bit):
    """SMILES of the substructure that sets a Morgan fingerprint bit, from the first molecule that has it."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    for mol in mols:
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        info = ao.GetBitInfoMap().get(bit)
        if not info:
            continue
        atom, radius = info[0]
        if radius == 0:
            return mol.GetAtomWithIdx(atom).GetSymbol()
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
        atoms = {atom}
        for b in env:
            bond = mol.GetBondWithIdx(b)
            atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=list(env))
    return "?"


def readable_names(mols, names, idx):
    """Label Morgan bits with the fragment they encode, e.g. 'morgan_80 [C(=O)O]'."""
    out = list(names)
    for i in idx:
        if names[i].startswith("morgan_"):
            out[i] = f"{names[i]} [{morgan_bit_fragment(mols, int(names[i].split('_')[1]))}]"
    return out


def main():
    model = joblib.load(MODEL)
    df = pd.read_csv(DATA)
    df["mol"] = df["smiles"].map(smiles_to_mol)
    df = df.dropna(subset=["mol"]).sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
    X = featurize_many(df["mol"])
    names = feature_names()

    explainer = shap.TreeExplainer(model)
    sv = explainer(X, check_additivity=False)[..., 1]  # class 1 = BBB-penetrating

    mean_abs = np.abs(sv.values).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:20]
    sv.feature_names = readable_names(df["mol"], names, top)

    pd.DataFrame({
        "feature": [sv.feature_names[i] for i in top],
        "mean_abs_shap": mean_abs[top].round(5),
        # Sign of the correlation between feature value and SHAP: + means "more of it helps BBB penetration"
        "direction": [
            "+" if np.corrcoef(X[:, i], sv.values[:, i])[0, 1] > 0 else "-"
            for i in top
        ],
    }).to_csv(OUT / "shap_top_features.csv", index=False)

    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title("SHAP: impact on P(crosses BBB)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.plots.bar(sv, max_display=20, show=False)
    plt.title("Mean |SHAP| (global importance)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    for dep in ["TPSA", "MolLogP"]:
        shap.plots.scatter(sv[:, names.index(dep)], color=sv, show=False)
        plt.tight_layout()
        plt.savefig(OUT / f"shap_dependence_{dep}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Per-molecule explanations for a brain-penetrant and a non-penetrant drug.
    ex_mols = [smiles_to_mol(s) for s in EXAMPLES.values()]
    ex = explainer(featurize_many(ex_mols), check_additivity=False)[..., 1]
    ex_top = np.argsort(np.abs(ex.values).sum(axis=0))[::-1][:30]
    ex.feature_names = readable_names(ex_mols, names, ex_top)
    for i, name in enumerate(EXAMPLES):
        p = model.predict_proba(featurize_many([ex_mols[i]]))[0, 1]
        shap.plots.waterfall(ex[i], max_display=12, show=False)
        plt.title(f"{name}: P(crosses BBB) = {p:.2f}")
        plt.tight_layout()
        plt.savefig(OUT / f"shap_waterfall_{name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{name}: P(BBB+) = {p:.2f}")

    print(pd.read_csv(OUT / "shap_top_features.csv").to_string(index=False))
    print(f"Saved SHAP plots to {OUT}")


if __name__ == "__main__":
    main()
