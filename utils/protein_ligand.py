import numpy as np
from ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from moleculekit.smallmol.smallmol import SmallMol
from ecloud_utils.grid import rotate, BuildGridCenters, generate_sigmas
from moleculekit.util import uniformRandomRotation
from moleculekit.tools.voxeldescriptors import _getOccupancyC
from utils.chem import set_mol_position, read_sdf, write_pkl

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

