import pickle
from rdkit import Chem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import copy
try:
    from .pdb_parser import PDBProtein
except:
    from utils.pdb_parser import PDBProtein

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


def pocket_trunction(pdb_file, threshold=10, outname=None, sdf_file=None, centroid=None):
    
    pdb_parser = PDBProtein(pdb_file)
    if centroid is None:
        centroid = sdf2centroid(sdf_file)
    else:
        centroid = centroid
    residues = pdb_parser.query_residues_radius(centroid,threshold)
    residue_block = pdb_parser.residues_to_pdb_block(residues)
    if outname is None:
        outname = pdb_file[:-4]+f'_pocket{threshold}.pdb'
    f = open(outname,'w')
    f.write(residue_block)
    f.close()

    return outname
def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z
