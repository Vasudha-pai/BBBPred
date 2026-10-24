"""Turn SMILES strings into numeric feature vectors for the random forest."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
FP_BITS = 2048
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
_largest_fragment = rdMolStandardize.LargestFragmentChooser()


def smiles_to_mol(smiles):
    """Parse a SMILES string and strip salts/counter-ions. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _largest_fragment.choose(mol)


def scaffold(mol):
    """Bemis-Murcko scaffold, used to group structurally similar molecules."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def featurize(mol):
    """RDKit physico-chemical descriptors + Morgan (ECFP4) fingerprint bits."""
    desc = np.array(list(Descriptors.CalcMolDescriptors(mol).values()), dtype=np.float64)
    fp = _morgan.GetFingerprintAsNumPy(mol).astype(np.float64)
    return np.concatenate([desc, fp])


def feature_names():
    return DESCRIPTOR_NAMES + [f"morgan_{i}" for i in range(FP_BITS)]


def featurize_many(mols):
    X = np.vstack([featurize(m) for m in mols])
    # A few descriptors can overflow on unusual molecules; keep values finite.
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6)
