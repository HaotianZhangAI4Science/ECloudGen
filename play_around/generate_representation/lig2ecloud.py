import os
import pandas as pd
import numpy as np
from grid2 import BuildGridCenters
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import h5py
from xtb_density import interplot_ecloud
from chem import get_geom, get_center, read_sdf
from glob import glob
import shutil
# Now choose the protocol for your data, size, N, respolution control the grid size
def protocol(mode=32):
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

def get_ligecloud(mol,calculater, protocol, add_noise=True):
    '''
    Input:
        mol: rdkit 3D mol
        calculater: xtb density calculater
        protocol: protocol for the grid, format: {'grids':(32, 32, 32, 3), 'N':[32, 32, 32]}
        add_noise: add noise to the ligand grid
    Output:
        lig_density: ligand electron density, shape: (32, 32, 32)
    '''
    stand_grid = protocol['grids']
    N = protocol['N']
    mol_center = mol_center = get_center(mol) 
    lig_grids = stand_grid + mol_center 
    if add_noise:
        lig_grids += np.random.randn(3).astype(np.float32)
    lig_ecloud = calculater.calculate(mol)
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(N)
    return lig_density

if __name__ == '__main__':
    # read ligand sdf
    ligand = read_sdf('ligand.sdf')[0]
    # read xtb calculater
    from xtb_density import CDCalculator
    claculator = CDCalculator(xtb_command='xtb')
    # get ligand density
    lig_density = get_ligecloud(ligand, claculator, protocol())

    # remove the temp folder
    temp_dirs = glob('./temp/*')
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)
        print(f'{temp_dir} removed')

    # save ligand density
    np.save(f'./data/ecloud/lig_density.npy',lig_density)
    print('Ligand density saved as lig_density.npy')