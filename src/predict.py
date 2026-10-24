"""Predict BBB penetration for new molecules.

Usage: python src/predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "CC(=O)Oc1ccccc1C(=O)O"
"""
import sys
from pathlib import Path

import joblib

from features import featurize_many, smiles_to_mol

MODEL = Path(__file__).resolve().parent.parent / "outputs" / "rf_bbbp.joblib"


def main(smiles_list):
    model = joblib.load(MODEL)
    for smi in smiles_list:
        mol = smiles_to_mol(smi)
        if mol is None:
            print(f"{smi}\tinvalid SMILES")
            continue
        p = model.predict_proba(featurize_many([mol]))[0, 1]
        verdict = "penetrates" if p >= 0.5 else "does not penetrate"
        print(f"{smi}\tP(BBB+) = {p:.2f}\t{verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
