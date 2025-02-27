import pickle
from rdkit import Chem
from .sascorer import compute_sa_score as sa

def qed(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    return Chem.QED.qed(mol)

def logp(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    return Chem.Crippen.MolLogP(mol)

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(i)
    writer.close()

def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(list,file):
    with open(file,'wb') as f:
        pickle.dump(list,f)
        print('pkl file saved at {}'.format(file))

def read_txt(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            result.append(line.strip())
    return result

import copy
def rm_radical(mol):
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol

def rdkit_normal_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            normal_smi = Chem.MolToSmiles(mol)
            return normal_smi
    except Exception as e:
        print(f"Error processing SMILES '{smi}': {e}")
    return None

def token_clean(smi, tokenizer):
    try:
        if smi != "*":
            tokenizer.tokenize_text("[SMILES]" + smi + "[STOP]", pad=True)
            return smi
    except Exception as e:
        print(f"Error tokenize SMILES '{smi}'")
    return None

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def remove_atom_indices(mol):
    """
    Removes atom mapping numbers set by mol_with_atom_index function from a molecule.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule object with atom indices set.

    Returns:
    rdkit.Chem.Mol: A molecule object with removed atom indices.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return mol

def set_double_bond(mol, idx1, idx2):
    """
    Sets the bond between two specified atoms in a molecule to a double bond.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule object.
    idx1 (int): Index of the first atom.
    idx2 (int): Index of the second atom.

    Returns:
    rdkit.Chem.Mol: A new molecule object with the modified bond.
    """
    new_mol = Chem.RWMol(mol)
    bond = new_mol.GetBondBetweenAtoms(idx1, idx2)
    if bond is None:
        raise ValueError("No bond exists between the specified atoms.")
    bond.SetBondType(Chem.BondType.DOUBLE)

    return new_mol.GetMol()