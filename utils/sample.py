from rdkit import Chem
import torch
import os.path as osp
from .grid import voxelize_mol
from .dataset import vocab_list, vocab_i2c, vocab_c2i


def transform_smiles(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids

def decode_smiles(in_tensor):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c[xchar]
        csmile = csmile.replace('X','Cl').replace('Y','[nH]').replace('Z','Br')
        csmile = csmile.replace('L','[N+]').replace('M','[O-]').replace('Q','[n+]')
        gen_smiles.append(csmile)
    return gen_smiles
    

class decipher:

    def __init__(self, model, ckpt_path):
        super(decipher,self).__init__()

        self.model = model
        self.ckpt_path = ckpt_path
        
    def load_model(self):
        
        ckpt = torch.load(self.ckpt_path)
        self.model.load_state_dict(ckpt['model'])

        return model

    def decipher_mol(self,mol,prop=False, n_attemps=20):
        
        vox = voxelize_mol(mol)
        vox = torch.tensor(vox).repeat(n_attemps,1,1,1,1)
        shape_emb = self.model.encoder(vox)
        if prop:
            smiles = self.model.decoder.sample_prob(shape_emb)
        else:
            smiles = self.model.decoder.sample(shape_emb)
        smiles = torch.stack(smiles,1).data.numpy()
        smiles = decode_smiles(smiles)
        smiles = transform_smiles(smiles)

        return smiles
    
    def decipher_ecloud(self,ecloud,prop=False,n_attemps=20):
        
        if type(ecloud) == torch.Tensor:
            vox = ecloud
            vox = vox.repeat(n_attemps,1,1,1,1)
        else:
            vox = torch.tensor(ecloud)
            vox = vox.repeat(n_attemps,1,1,1,1)
        shape_emb = self.model.encoder(vox)
        if prop:
            smiles = self.model.decoder.sample_prob(shape_emb)
        else:
            smiles = self.model.decoder.sample(shape_emb)
        smiles = torch.stack(smiles,1).data.numpy()
        smiles = decode_smiles(smiles)
        smiles = transform_smiles(smiles)

        return smiles