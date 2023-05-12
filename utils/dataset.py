import torch
from torch.utils.data import Dataset, DataLoader
from moleculekit.smallmol.smallmol import SmallMol
from rdkit import Chem
import pickle
import os.path as osp
from rdkit.Chem import AllChem

def remove_chirality(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Loop through all atoms in the molecule
    for atom in mol.GetAtoms():
        # Clear chiral information for the atom
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        
    # Convert the modified Mol object back to a SMILES string
    new_smiles = Chem.MolToSmiles(mol)
    
    return new_smiles
    
def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

try:
    from .grid import *
except:
    from utils.grid import *

vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "X", "Y", "Z", # "Cl", "[nH]", "Br"
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",  # Misc,
              'L','Q','M'#"[N+]","[n+]",'[O-]'
]
vocab_i2c = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i = {vocab_i2c[i]: i for i in vocab_i2c}

def transform_vox(pkt_mol, lig_mol):
    pkt_mol, lig_mol = SmallMol(pkt_mol), SmallMol(lig_mol)
    pkt_vox, lig_vox = vox_from_pair(pkt_mol, lig_mol)
    pkt_vox, lig_vox = torch.tensor(pkt_vox), torch.tensor(lig_vox)
    return (pkt_vox,lig_vox)

class ECloud(Dataset):
    
    def __init__(self, data_pairs,data_base, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
        self.data_base = data_base
    
    def __len__(self):
        return len(self.data_pairs)
        
    def __getitem__(self, idx):
        pkt_fn, lig_fn = self.data_pairs[int(idx)][0], self.data_pairs[int(idx)][1]
        lig_mol = Chem.MolFromMolFile(osp.join(self.data_base,lig_fn))
        pkt_mol = Chem.MolFromPDBFile(osp.join(self.data_base,pkt_fn))
        try:
            pkt_vox,lig_vox = transform_vox(lig_mol, pkt_mol)
        except Exception as e:
            print(e)
            pkt_vox,lig_vox = self.__getitem__(int(idx)+1)

        return pkt_vox,lig_vox


class ECloudSMI(Dataset):
    '''
    This calss could also be used the translator from smiles string to the numerical representation
    e.g. translator = ECloudSMI()
    smi = 'CCNOCC'
    emb = translator.str2emb('CCNOCC')
    smi = translator.emb2str(emb)
    '''
    def __init__(self, mol_paths=None, mol_list=None, data_base=None, smi_list=None):
        self.mol_list = mol_list
        self.mol_paths = mol_paths
        self.smi_list = smi_list
        self.data_base = data_base
        self.vocab_i2c = {i: x for i, x in enumerate(vocab_list)}
        self.vocab_c2i = {self.vocab_i2c[i]: i for i in self.vocab_i2c}

    def __len__(self):
        if self.mol_list is not None:
            data_len = len(self.mol_list)
        if self.mol_paths is not None:
            data_len = len(self.mol_paths)
        if self.smi_list is not None:
            data_len = len(self.smi_list)
        return data_len

    def __getitem__(self, index):
        if self.mol_list is not None:
            mol = self.mol_list[index]
        if self.mol_paths:
            try:
                mol = Chem.MolFromMolFile(osp.join(self.data_base,self.mol_paths[index]))
            except Exception as e:
                print('read fails: ',e)
                return (None, None, None)
                
        if self.smi_list:
            try:
                mol = Chem.MolFromSmiles(self.smi_list[index])
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except Exception as e:
                print('Confgeneration fails: ',e, self.smi_list[index])
                return (None, None, None)

        try:
            Chem.RemoveStereochemistry(mol)
            mol = Chem.RemoveHs(mol)
            smi = Chem.MolToSmiles(mol)
            smile = self.str2emb(smi)
            smi, length = self.process_smiemb(smile)
            vox = voxelize_mol(mol)
        except Exception as e:
            print('processing fails: ',e, index)
            return (None, None, None)
        
        return (torch.tensor(vox), smi, length)

    def process_smiemb(self,smiemb):
        smiles_ = list(smiemb)
        end_token = smiles_.index(2) # 2 is the end token
        return torch.Tensor(smiemb), end_token+1 # end_token+1 is the length of the smiles
    
    def str2emb(self, str, cananical=True):
        # rdkit cananical
        if cananical:
            mol = Chem.MolFromSmiles(str)
            str = Chem.MolToSmiles(mol)
        str = str.replace('Cl','X').replace('[nH]','Y').replace('Br','Z')
        str = str.replace('[SH]','S').replace('[N+]','L').replace('[NH+]','L').replace('[O-]','M').replace('[n+]','Q')
        emb = [1] + [self.vocab_c2i[xchar] for xchar in str] + [2]
        return emb

    def emb2str(self,emb):
        smiles = list(emb)
        end_token = smiles.index(2)
        smile_str = "".join([self.vocab_i2c[i] for i in smiles[1:end_token]])
        return smile_str


def ECloudSMI_collate_fn(batch):

    batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]
    
    batch.sort(key=lambda x: x[2], reverse=True) # (smi, length)
    voxes, smiles, lengths = zip(*batch)

    voxes = torch.stack(voxes, 0)

    smi_targets = torch.zeros(len(smiles), max(lengths)).long()

    for i, smile in enumerate(smiles):
        end = lengths[i]
        smi_targets[i, :end] = smile[:end]
    
    return voxes, smi_targets, lengths


def get_dataset(split_file, data_base, use_cache=None):
    '''
    This function is deprecated because the memory explodes when all the data is loaded
    But if you have tiny dataset, (~1000-10000 mols), I highly recommend you use this.
    The db file storage form is quit confusing for freshman
    '''

    if use_cache:
        train_set = read_pkl('./data/train_pairs.pkl')
        test_set = read_pkl('./data/test_pairs.pkl')
    else:
        split = torch.load(split_file)
        train_set = []
        test_set = []
        for idx, (pkt_fn, lig_fn) in enumerate(tqdm(split['train'])):
            try:
                lig_mol = Chem.MolFromMolFile(osp.join(data_base,lig_fn))
                pkt_mol = Chem.MolFromPDBFile(osp.join(data_base,pkt_fn))
                if (lig_mol is not None) and (pkt_mol is not None):
                    train_set.append((pkt_mol, lig_mol))
            except Exception as e:
                print(e)
                ... 

        for idx, (pkt_fn, lig_fn) in enumerate(split['test']):
            try:
                lig_mol = Chem.MolFromMolFile(osp.join(data_base,lig_fn))
                pkt_mol = Chem.MolFromPDBFile(osp.join(data_base,pkt_fn))
                if (lig_mol is not None) and (pkt_mol is not None):
                    test_set.append((pkt_mol, lig_mol))
            except:
                ... 
    return train_set, test_set

