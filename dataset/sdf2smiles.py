import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
import argparse
RDLogger.DisableLog('rdApp.*')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_root', type=str, default='/mnt/d/KunLai/LDM3D/ECloud_data/sdf/')
    parser.add_argument('--save_file', type=str, default='/mnt/d/KunLai/LDM3D/ECloud_data/smiles.txt')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    sdf_root = args.sdf_root
    save_file = args.save_file

    files = os.listdir(sdf_root)
    num_files = len(files)
    with open(save_file, 'w') as f:
        for i in tqdm(range(num_files)):
            file_name = os.path.join(sdf_root, str(i) + '.sdf')
            mol = Chem.MolFromMolFile(file_name)
            smiles = Chem.MolToSmiles(mol)
            f.write(smiles + '\n')
