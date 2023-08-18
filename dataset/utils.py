from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import Chem

def calculate_hbd(mol):
    return Chem.rdMolDescriptors.CalcNumHBA(mol)

def calculate_hba(mol):
    return Chem.rdMolDescriptors.CalcNumHBD(mol)

def calculate_tpsa(mol):
    return Chem.rdMolDescriptors.CalcTPSA(mol)

def calculate_mw(mol):
    return Descriptors.MolWt(mol)