from rdkit import Chem
from rdkit.Chem import AllChem

def get_geom(mol, mmff=False):
    mol_ = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_)
    if mmff:
        AllChem.MMFFOptimizeMolecule(mol_)
    mol_ = Chem.RemoveHs(mol_)
    return mol_

def get_center(mol):
    return mol.GetConformer().GetPositions().mean(axis=0)

def read_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    return [mol for mol in suppl]