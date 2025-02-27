# pickle files are the trained QSAR models, adopted from https://github.com/mims-harvard/TDC 
# specific version of sklearn is required 
# conda install scikit-learn=1.2.2
import os
import pickle
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
import numpy as np
current_script_path = os.path.abspath(__file__)
directory_of_script = os.path.dirname(current_script_path)

class qsar_model():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, target_name):
        if target_name == "drd2":
            self.clf_path = os.path.join(directory_of_script, "drd2.pkl")
        elif target_name == "gsk3":
            self.clf_path = os.path.join(directory_of_script, "gsk3.pkl")
        elif target_name == "jnk3":
            self.clf_path = os.path.join(directory_of_script, "jnk3.pkl")

        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol):
        if mol is not None:
            fp = qsar_model.fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1][0]
            return np.float32(score)
        else:
            raise ValueError("Molecule is None")
            # return np.float32(0) 

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)