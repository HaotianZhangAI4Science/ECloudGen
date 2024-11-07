import numpy as np
import pickle
import argparse
import lmdb
import os
import logging
import os.path as osp
import h5py
logging.getLogger("moleculekit").setLevel(logging.WARNING)
from rdkit import Chem
from ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from ecloud_utils.grid import rotate, BuildGridCenters, generate_sigmas
from utils.chem import set_mol_position, read_sdf, write_pkl
from moleculekit.util import uniformRandomRotation
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import _getOccupancyC
from tqdm import tqdm
from multiprocessing import Pool
# from utils.protein_ligand import get_occu_ecloud_pair, protocol


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

def write_lmdb(output_dir, name):

    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f'{name}.lmdb')

    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn_write = env_new.begin(write=True)

    return txn_write, env_new

# calculater = CDCalculator(xtb_command='xtb')
# or you can use definite path of the xtb
def get_occu_ecloud_pair(pkt_mol, lig_mol, grid_protocol):
    calculater = CDCalculator(xtb_command='xtb')
    pkt_smallmol = SmallMol(pkt_mol)

    lig_coords = lig_mol.GetConformer().GetPositions()
    lig_center = lig_coords.mean(axis=0)

    # define the pkt channel
    pkt_sigmas, pkt_coords, pkt_center = generate_sigmas(pkt_smallmol)

    # use the pkt_center as the whole center
    pkt_grids = grid_protocol['grids'] + pkt_center
    lig_grids = grid_protocol['grids'] + pkt_center

    # Do the random rotation
    rrot = uniformRandomRotation()  # Rotation
    lig_coords = rotate(lig_coords, rrot, center=pkt_center)
    # pkt_coords_ = rotate(pkt_mol.GetConformer().GetPositions(), rrot, center=pkt_center)
    pkt_coords = rotate(pkt_coords, rrot, center=pkt_center)
    size = grid_protocol['N'][0]
    # use VdW occupancy to represent the pkt channel
    pkt_channel = _getOccupancyC(pkt_coords.astype(np.float32),
                                 pkt_grids.reshape(-1, 3),
                                 pkt_sigmas).reshape(size, size, size, 5)
    # set the ligand position for the ecloud calculation
    rotated_lig_mol = set_mol_position(lig_mol, lig_coords)
    lig_ecloud = calculater.calculate(rotated_lig_mol)

    # interplot the ecloud to the grid, manually transform the ligand grid coordinate and its densitu to the pkt grid
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(grid_protocol['N'])

    return pkt_channel, lig_density


def single_process(index):
    '''
    Input:
        index: a tuple of pdb_index and sdf_index
    Output:
        {'pocket': pkt_channel_list, 'ligand': lig_density_list}
        
    Note: default grid protocol is 64
    Note: I use the random data enhancement five times to softly embed rotational equivariance into the model.
          The group-equivariant CNN (Taco Cohen et al.) can be used to replace the data enhancement, but they 
          basically perform random rotation at the embedding space, which is GPU-memory entensive. 
          And it does not show superior performance when comparing with the data enhancement approach (at least from my perspective).
    '''
    try:
        pdb_index, sdf_index = index[:2]
        pdb_path = os.path.join(root, pdb_index)
        sdf_path = os.path.join(root, sdf_index)
        pkt_mol = Chem.MolFromPDBFile(pdb_path)
        lig_mol = read_sdf(sdf_path)[0]
        pkt_channel_list = []
        lig_density_list = []
        for _ in range(5): 
            pkt_channel, lig_density = get_occu_ecloud_pair(pkt_mol, lig_mol, grid_protocol=protocol(32))
            pkt_channel = pkt_channel.astype(np.float16)
            lig_density = lig_density.astype(np.float16)
            pkt_channel_list.append(pkt_channel)
            lig_density_list.append(lig_density)
        return {'pocket': pkt_channel_list, 'ligand': lig_density_list}
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data_process ")
    parser.add_argument("--root", type=str, default="./data/crossdocked_pocket10")
    parser.add_argument("--index_path", type=str, default="./data/crossdocked_pocket10/index.pkl")
    parser.add_argument("--save_path", type=str, default="../data")
    parser.add_argument("--cache_ecloud", type=str, default='../data/cache')
    parser.add_argument("--mode", type=str, default='valid')
    parser.add_argument("--processor", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100, help="None for all data")
    args = parser.parse_args()

    root = args.root
    with open(args.index_path, 'rb') as f:
        indexs = pickle.load(f)
    total = len(indexs)

    print(f"############# Processing {total} Protein-Ligand Pairs #############")
    if args.mode == 'train':
        if args.end is None:
            args.end = len(indexs[1000:])
        indexs = indexs[1000:][args.start: args.end]
    else:
        if args.end is None:
            args.end = len(indexs[:1000])
        indexs = indexs[:1000][args.start: args.end]
    h5_path = osp.join(args.save_path, 'pkt_lig_' + args.mode + '_' + str(args.start) + '_' + str(args.end)+'.h5')
    nums = args.end
    success_indexs = []

    with Pool(processes=args.processor) as pool:
        iters = pool.imap(single_process, indexs)
        for i, mol_dict in tqdm(enumerate(iters), total=len(indexs)):
            if mol_dict is not None:
                try:
                    protein_npy = indexs[i][0].replace('.pdb', '.npy')
                    ligand_npy = indexs[i][1].replace('.sdf', '.npy')
                    cache_dir = osp.dirname(os.path.join(args.cache_ecloud, protein_npy))
                    os.makedirs(cache_dir, exist_ok=True)
                    ecloud_pocket = np.array(mol_dict['pocket'])[:,:,:,:,4]
                    ecloud_ligand = np.array(mol_dict['ligand'])
                    np.save(os.path.join(args.cache_ecloud, protein_npy), ecloud_pocket)
                    np.save(os.path.join(args.cache_ecloud, ligand_npy), ecloud_ligand)
                    success_indexs.append((protein_npy, ligand_npy, indexs[i][2]))
                except Exception as e:
                    print('Error Occured: {}'.format(e))
                    continue
    write_pkl(success_indexs, osp.join(args.save_path, 'success_indexs.pkl'))
    print('Generate successful index at {}'.format(osp.join(args.save_path, 'success_indexs.pkl')))

    nums = len(success_indexs)
    
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset(
            name='ecloud_pocket',
            shape=(nums, 5, 32, 32, 32),
            dtype=np.float16)
        hf.create_dataset(
            name='ecloud_ligand',
            shape=(nums, 5, 32, 32, 32), # change the 32, 32, 32 to your own size
            dtype=np.float16)
        for i, (protein_ecloud, ligand_ecloud, _) in enumerate(success_indexs):
            protein_ecloud = np.load(f'{args.cache_ecloud}/{protein_ecloud}')
            ligand_ecloud = np.load(f'{args.cache_ecloud}/{ligand_ecloud}')
            hf['ecloud_pocket'][i] = protein_ecloud
            hf['ecloud_ligand'][i] = ligand_ecloud

    print('Successfully processed', nums, 'pairs')
    print('processed h5 saved to', h5_path)
