import sys
sys.path.append('..')
import h5py
from rdkit import Chem
from models.ECloudDecipher.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from models.ECloudDecipher.models.encoding.tokenizers import get_vocab
from tqdm import tqdm
import argparse
from utils.chem import gen_geom_with_rdkit, set_mol_position
from ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from moleculekit.util import uniformRandomRotation
from ecloud_utils.grid import rotate, BuildGridCenters, generate_sigmas
import numpy as np
import multiprocessing

def protocol(mode=32):
    '''
    Define the grid grid_protocol, including grid size, resolution, and grid centers
        grid size: 32 or 64
        resolution: 0.5 or 0.2
        grid centers: the center of the grid
    Input:
        mode: grid mode, 32 or 64
    Output:
        {'grids':grids, 'N':N}
    '''
    size = mode
    N = [size, size, size]
    if mode == 32:
        resolution = 0.5
        llc = (np.zeros(3) - float(size * resolution / 2)) + resolution / 2
        grids = BuildGridCenters(llc, N, resolution)
    elif mode == 64:
        resolution = 0.2
        llc = (np.zeros(3) - float(size * resolution / 2)) + resolution / 2
        grids = BuildGridCenters(llc, N, resolution)
    
    return {'grids':grids, 'N':N}

def get_ecloud(smi, rand_rotate=True, grid_protocol=protocol()):
    calculater = CDCalculator(xtb_command='xtb')
    lig_mol = Chem.MolFromSmiles(smi)
    lig_mol = gen_geom_with_rdkit(lig_mol)
    lig_coords = lig_mol.GetConformer().GetPositions()
    lig_center = lig_coords.mean(axis=0)
    if rand_rotate:
        rrot = uniformRandomRotation()  # Rotation
        lig_coords = rotate(lig_coords, rrot, center=lig_center)
    rotated_lig_mol = set_mol_position(lig_mol, lig_coords)
    lig_ecloud = calculater.calculate(rotated_lig_mol)

    lig_grids = grid_protocol['grids'] + lig_center
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(grid_protocol['N'])
    return lig_density

def worker(index):
    smi = smiles[index]
    try:
        ecloud = get_ecloud(smi)
    except Exception as e:
        print(e)
        ecloud = None
    return index, smi, ecloud

def process_smiles_parallel(smiles_list, num_workers=8):
    successful_smiles = []
    eclouds = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker, range(len(smiles_list))), total=len(smiles_list)))

    # Collect successful results
    for index, smi, ecloud in results:
        if ecloud is not None:
            successful_smiles.append(smi)
            eclouds.append(ecloud)

    return successful_smiles, eclouds


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--smiles', type=str, default='data/demo.smi')
    args.add_argument('--output', type=str, default='data/ecloud_decipher.h5')
    args.add_argument('--num_workers', type=int, default=8)
    args = args.parse_args()

    with open(args.smiles) as f:
        smiles=[line.strip('\n') for line in f]
    print('demo_data len: ', len(smiles))

    successful_smiles, eclouds = process_smiles_parallel(smiles, num_workers=args.num_workers)

    with h5py.File(args.output, 'w') as f:
        f.create_dataset("smiles", data=np.array(successful_smiles, dtype='S'))  # Store as byte strings
        f.create_dataset("eclouds", data=np.array(eclouds))  # Store as a NumPy array

    print(f"Successfully saved {len(successful_smiles)} processed molecules to {args.output}")


