import pickle
from rdkit import Chem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import copy
def set_mol_position(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def remove_chirality(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Loop through all atoms in the molecule
    for atom in mol.GetAtoms():
        # Clear chiral information for the atom
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        
    # Convert the modified Mol object back to a SMILES string
    new_smiles = Chem.MolToSmiles(mol)
    
    return new_smiles
    
def align_pkt_lig_to_zero(lig_mol, pkt_mol):
    '''
    Align the pkt and lig mols to the pkt zero point
    Test Code
    lig_coords = lig_mol.GetConformer(0).GetPositions()
    lig_coords.mean(axis=0)
    '''
    lig_coords = lig_mol.GetConformer(0).GetPositions()
    pkt_coords = pkt_mol.GetConformer(0).GetPositions()
    lig_coords -= pkt_coords.mean(axis=0)
    pkt_coords -= pkt_coords.mean(axis=0)
    lig_mol = set_mol_position(lig_mol, lig_coords)
    pkt_mol = set_mol_position(pkt_mol, pkt_coords)
    return lig_mol, pkt_mol

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