"""Deeper analysis of where the model works and where it fails. Run after evaluate.py.

1. Applicability domain - is the model less reliable on molecules unlike its training data?
2. Error analysis       - the most confident mistakes, and the SHAP reasons behind them
3. Label consistency    - near-identical molecules with opposite labels (noise / activity cliffs)
4. Case studies         - well-known drugs (held out from training) with documented BBB behaviour
5. Chemical space       - t-SNE map of the dataset by label and by split

Usage: python src/analysis.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score

from data import OUT, SEED, load_split, make_rf
from features import DESCRIPTOR_NAMES, FP_BITS, feature_names, featurize_many, smiles_to_mol

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIM_BINS = [0, 0.3, 0.4, 0.5, 0.7, 1.01]
N_ERRORS = 8

# Expected behaviour from the pharmacology literature, with the mechanism that explains it.
CASE_STUDIES = [
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", 1, "Lipophilic sedative, passive diffusion"),
    ("Fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1", 1, "Antidepressant, passive diffusion"),
    ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1", 1, "1st-gen antihistamine (sedating)"),
    ("Fexofenadine", "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1", 0,
     "2nd-gen antihistamine: zwitterion + P-gp efflux"),
    ("Hydroxyzine", "OCCOCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1", 1, "1st-gen antihistamine (sedating)"),
    ("Cetirizine", "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1", 0,
     "Hydroxyzine metabolite: added acid limits entry"),
    ("Loperamide", "CN(C)C(=O)C(CCN1CCC(O)(CC1)c1ccc(Cl)cc1)(c1ccccc1)c1ccccc1", 0,
     "Opioid kept out by P-gp efflux"),
    ("Domperidone", "O=C1Nc2ccccc2N1CCCN1CCC(CC1)n1c(=O)[nH]c2cc(Cl)ccc21", 0,
     "Antiemetic kept out by P-gp efflux"),
    ("Dopamine", "NCCc1ccc(O)c(O)c1", 0, "Polar neurotransmitter"),
    ("L-DOPA", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O", 1, "Polar, but carried in by the LAT1 transporter"),
    ("Gabapentin", "NCC1(CC(=O)O)CCCCC1", 1, "Amino acid, carried in by LAT1"),
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", 0, "Hydrophilic beta-blocker (few CNS effects)"),
]

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def fingerprints(mols):
    return [_morgan.GetFingerprint(m) for m in mols]


def nearest_neighbours(query_fps, ref_fps):
    """Max Tanimoto similarity to the reference set and the index of that neighbour."""
    sims = np.array([DataStructs.BulkTanimotoSimilarity(fp, ref_fps) for fp in query_fps])
    return sims.max(axis=1), sims.argmax(axis=1)


def top_reasons(explainer, X, names, k=3):
    sv = explainer(X, check_additivity=False)[..., 1].values
    out = []
    for row in sv:
        idx = np.argsort(np.abs(row))[::-1][:k]
        out.append("; ".join(f"{names[i]} {row[i]:+.3f}" for i in idx))
    return out


def applicability_domain(y_te, proba, sim, threshold):
    bins = pd.cut(sim, SIM_BINS, right=False)
    rows = []
    for b, g in pd.Series(range(len(sim))).groupby(bins, observed=True):
        idx = g.to_numpy()
        yy, pp = y_te[idx], proba[idx]
        rows.append({
            "similarity_to_train": f"{b.left:.1f}-{min(b.right, 1):.1f}",
            "n": len(idx),
            "frac_bbb+": round(yy.mean(), 2),
            "roc_auc": round(roc_auc_score(yy, pp), 3) if 0 < yy.mean() < 1 else None,
            "accuracy": round(accuracy_score(yy, pp >= threshold), 3),
            "mean_abs_error": round(float(np.abs(yy - pp).mean()), 3),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "applicability_domain.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.hist(sim, bins=30, color="steelblue")
    ax1.set_xlabel("Max Tanimoto similarity to training set")
    ax1.set_ylabel("Test molecules")
    ax1.set_title("How novel is the test set?")
    x = np.arange(len(table))
    ax2.bar(x, table["roc_auc"].fillna(0), color="steelblue", label="ROC-AUC")
    ax2.plot(x, table["accuracy"], "o-", color="darkorange", label="Accuracy (tuned thr.)")
    ax2.set_xticks(x, [f"{r}\n(n={n})" for r, n in zip(table["similarity_to_train"], table["n"])],
                   fontsize=8)
    ax2.set_ylim(0.5, 1)
    ax2.set_xlabel("Similarity to nearest training molecule")
    ax2.set_title("Performance vs. novelty")
    ax2.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "applicability_domain.png", dpi=150)
    plt.close()
    return table


def error_analysis(df, X, y, tr, te, proba, sim, nn, threshold, explainer, names):
    test = df.iloc[te].reset_index(drop=True)
    test["label"] = y[te]
    test["p_bbb"] = proba
    test["sim_to_train"] = sim
    test["nearest_train"] = df["name"].to_numpy()[tr][nn]
    test["nearest_train_label"] = y[tr][nn]
    for d in ["TPSA", "MolLogP", "MolWt", "NumHDonors"]:
        test[d] = X[te][:, DESCRIPTOR_NAMES.index(d)].round(1)
    fp = test[(test["label"] == 0) & (test["p_bbb"] >= threshold)].nlargest(N_ERRORS, "p_bbb")
    fn = test[(test["label"] == 1) & (test["p_bbb"] < threshold)].nsmallest(N_ERRORS, "p_bbb")
    errors = pd.concat([fp.assign(error="false positive"), fn.assign(error="false negative")])
    errors["top_shap_reasons"] = top_reasons(explainer, X[te][errors.index], names)
    cols = ["error", "name", "smiles", "label", "p_bbb", "TPSA", "MolLogP", "MolWt", "NumHDonors",
            "sim_to_train", "nearest_train", "nearest_train_label", "top_shap_reasons"]
    errors = errors[cols].round({"p_bbb": 3, "sim_to_train": 2})
    errors.to_csv(OUT / "error_analysis.csv", index=False)

    img = Draw.MolsToGridImage(
        list(test.loc[errors.index, "mol"]), molsPerRow=4, subImgSize=(260, 220),
        legends=[f"{r['name'][:28]}\n{'FP' if r['error'] == 'false positive' else 'FN'}  P={r['p_bbb']:.2f}"
                 for _, r in errors.iterrows()])
    img.save(OUT / "error_molecules.png")

    # Do mistakes concentrate on unfamiliar chemistry?
    wrong = (test["p_bbb"] >= threshold).astype(int) != test["label"]
    return errors, {
        "mean_sim_correct": round(float(test.loc[~wrong, "sim_to_train"].mean()), 3),
        "mean_sim_wrong": round(float(test.loc[wrong, "sim_to_train"].mean()), 3),
        "fp_median_TPSA": float(test.loc[wrong & (test["label"] == 0), "TPSA"].median()),
        "fn_median_TPSA": float(test.loc[wrong & (test["label"] == 1), "TPSA"].median()),
    }


def label_consistency(df, y, fps, cutoff=0.9):
    """Pairs of (almost) identical molecules with opposite labels."""
    inchikey = [Chem.MolToInchiKey(m) for m in df["mol"]]
    rows = []
    for i in range(len(fps)):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        for j in np.where(sims >= cutoff)[0] + i + 1:
            if y[i] != y[j]:
                rows.append({"mol_a": df["name"][i], "label_a": y[i], "mol_b": df["name"][j],
                             "label_b": y[j], "tanimoto": round(float(sims[j - i - 1]), 3),
                             "identical_structure": inchikey[i] == inchikey[j],
                             "smiles_a": df["smiles"][i], "smiles_b": df["smiles"][j]})
    conflicts = pd.DataFrame(rows).sort_values("tanimoto", ascending=False)
    conflicts.to_csv(OUT / "label_conflicts.csv", index=False)
    return conflicts


def case_studies(df, X, y, tr, names, threshold):
    """Predict known drugs with a model that has never seen them.

    Most of these drugs are in BBBP, so the train-split model would just recall their
    labels. Instead we refit on the training split minus every case-study drug.
    """
    skeleton = np.array([Chem.MolToInchiKey(m)[:14] for m in df["mol"]])  # ignores stereo
    keys = {k: i for i, k in enumerate(skeleton)}
    mols = [smiles_to_mol(s) for _, s, _, _ in CASE_STUDIES]
    case_keys = {Chem.MolToInchiKey(m)[:14] for m in mols}
    keep = np.array([i for i in tr if skeleton[i] not in case_keys])
    print(f"  refitting without {len(tr) - len(keep)} case-study molecules")
    model = make_rf().fit(X[keep], y[keep])
    explainer = shap.TreeExplainer(model)
    Xc = featurize_many(mols)
    proba = model.predict_proba(Xc)[:, 1]
    rows = []
    for (name, smi, expected, note), m, p, reason in zip(CASE_STUDIES, mols, proba,
                                                       top_reasons(explainer, Xc, names, k=2)):
        idx = keys.get(Chem.MolToInchiKey(m)[:14])
        rows.append({"drug": name, "expected": "crosses" if expected else "blocked",
                     "p_bbb": round(float(p), 2),
                     "correct@0.5": bool((p >= 0.5) == expected),
                     "correct@tuned": bool((p >= threshold) == expected),
                     "bbbp_label": "-" if idx is None else int(df["p_np"][idx]),
                     "why": note, "top_shap_reasons": reason})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "case_studies.csv", index=False)

    plt.figure(figsize=(7, 5))
    colors = ["seagreen" if e == "crosses" else "indianred" for e in table["expected"]]
    plt.barh(table["drug"][::-1], table["p_bbb"][::-1], color=colors[::-1])
    plt.axvline(0.5, color="grey", ls=":", lw=1, label="0.5")
    plt.axvline(threshold, color="black", ls="--", lw=1, label=f"tuned threshold {threshold:.2f}")
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    for k, (p, ok) in enumerate(zip(table["p_bbb"][::-1], table["correct@tuned"][::-1])):
        plt.text(p + 0.01, k, "" if ok else "✗", va="center", color="black", fontsize=11)
    plt.xlim(0, 1.08)
    plt.xlabel("Predicted P(crosses BBB)")
    plt.title("Known drugs, held out from training\n(green = known to cross, red = blocked; ✗ = wrong at tuned threshold)",
              fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "case_studies.png", dpi=150)
    plt.close()
    return table


def chemical_space(fps, y, tr, te, proba, threshold):
    arr = np.array([list(fp) for fp in fps], dtype=bool)
    emb = TSNE(n_components=2, metric="jaccard", init="random", perplexity=30,
               random_state=SEED).fit_transform(arr)
    split = np.full(len(y), "train", dtype=object)
    split[te] = "test"
    wrong = np.zeros(len(y), bool)
    wrong[te] = (proba >= threshold).astype(int) != y[te]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for lab, col, txt in [(0, "indianred", "Blocked"), (1, "seagreen", "Crosses")]:
        axes[0].scatter(*emb[y == lab].T, s=5, c=col, alpha=0.6, label=txt)
    axes[0].set_title("Chemical space coloured by BBB label")
    axes[0].legend(markerscale=3)
    axes[1].scatter(*emb[split == "train"].T, s=5, c="lightgrey", label="Train")
    axes[1].scatter(*emb[(split == "test") & ~wrong].T, s=6, c="steelblue", label="Test, correct")
    axes[1].scatter(*emb[wrong].T, s=14, c="crimson", marker="x", label="Test, wrong")
    axes[1].set_title("Scaffold split: test chemotypes vs. training data")
    axes[1].legend(markerscale=2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("t-SNE of Morgan fingerprints (Jaccard / Tanimoto distance)")
    plt.tight_layout()
    plt.savefig(OUT / "chemical_space.png", dpi=150)
    plt.close()


def main():
    df, X, y, tr, te, model = load_split()
    eval_path = OUT / "evaluation.json"
    threshold = (json.loads(eval_path.read_text())["threshold"]["tuned_on_train_cv"]["threshold"]
                 if eval_path.exists() else 0.5)
    print(f"Decision threshold: {threshold}")
    proba = model.predict_proba(X[te])[:, 1]
    fps = fingerprints(df["mol"])
    sim, nn = nearest_neighbours([fps[i] for i in te], [fps[i] for i in tr])
    explainer = shap.TreeExplainer(model)
    names = feature_names()
    report = {"threshold": threshold}

    print("1. Applicability domain")
    ad = applicability_domain(y[te], proba, sim, threshold)
    print(ad.to_string(index=False))
    report["applicability_domain"] = ad.to_dict(orient="records")

    print("2. Error analysis")
    errors, summary = error_analysis(df, X, y, tr, te, proba, sim, nn, threshold, explainer, names)
    print(errors[["error", "name", "p_bbb", "TPSA", "MolLogP", "sim_to_train", "top_shap_reasons"]]
          .to_string(index=False))
    print(summary)
    report["errors"] = summary

    print("3. Label consistency")
    conflicts = label_consistency(df, y, fps)
    print(f"{len(conflicts)} pairs with Tanimoto >= 0.9 and opposite labels, "
          f"{int(conflicts['identical_structure'].sum())} of them the exact same structure")
    print(conflicts.head(10).drop(columns=["smiles_a", "smiles_b"]).to_string(index=False))
    report["label_conflicts_tanimoto_0.9"] = len(conflicts)
    report["label_conflicts_identical_structure"] = int(conflicts["identical_structure"].sum())

    print("4. Case studies")
    cases = case_studies(df, X, y, tr, names, threshold)
    print(cases.drop(columns="top_shap_reasons").to_string(index=False))
    report["case_studies_correct"] = {
        "at_0.5": f"{int(cases['correct@0.5'].sum())}/{len(cases)}",
        "at_tuned_threshold": f"{int(cases['correct@tuned'].sum())}/{len(cases)}"}

    print("5. Chemical space (t-SNE)")
    chemical_space(fps, y, tr, te, proba, threshold)

    (OUT / "analysis.json").write_text(json.dumps(report, indent=2))
    print(f"Saved analysis to {OUT}")


if __name__ == "__main__":
    main()
