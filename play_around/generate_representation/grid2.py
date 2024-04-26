import numpy as np
from moleculekit.tools.voxeldescriptors import _getOccupancyC
from moleculekit.util import uniformRandomRotation
from moleculekit.smallmol.smallmol import SmallMol
from rdkit import Chem
try:
    from htmd_utils2 import _getChannelRadii
except:
    from MG.ECloudGen_ELIP.utils.htmd_utils import _getChannelRadii

def BuildGridCenters(llc, N, step):
    """
    llc: lower left corner
    N: number of cells in each direction
    step: step size
    """

    if type(step) == float:
        xrange = [llc[0] + step * x for x in range(0, N[0])]
        yrange = [llc[1] + step * x for x in range(0, N[1])]
        zrange = [llc[2] + step * x for x in range(0, N[2])]
    elif type(step) == list or type(step) == tuple:
        xrange = [llc[0] + step[0] * x for x in range(0, N[0])]
        yrange = [llc[1] + step[1] * x for x in range(0, N[1])]
        zrange = [llc[2] + step[2] * x for x in range(0, N[2])]

    centers = np.zeros((N[0], N[1], N[2], 3))
    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])
    return centers

resolution = 1.
size = 24
N = [size, size, size]
llc = (np.zeros(3) - float(size * 1. / 2))
# Now, the box is 24×24×24 A^3
expanded_pcenters = BuildGridCenters(llc, N, resolution)


def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """

    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center


def get_aromatic_groups(in_mol):
    """
    Obtain groups of aromatic rings
    """
    groups = []
    ring_atoms = in_mol.GetRingInfo().AtomRings()
    for ring_group in ring_atoms:
        if all([in_mol.GetAtomWithIdx(x).GetIsAromatic() for x in ring_group]):
            groups.append(ring_group)
    return groups

def generate_occpy(mol):
    """
    Only Calculates the occupancy
    """
    coords = mol._coords[: , : , 0]
    n_atoms = len(coords)
    lig_center = mol.getCenter()
    

def generate_sigmas(mol):
    """
    Calculates sigmas for elements as well as pharmacophores.
    Returns sigmas, coordinates and center of ligand.
    """
    coords = mol._coords[: , : , 0]
    n_atoms = len(coords)
    lig_center = mol.getCenter()

    # Calculate all the channels
    multisigmas = _getChannelRadii(mol) 

    return multisigmas, coords, lig_center
    

def voxelize_pkt_lig(dual_sigmas, dual_coords, dual_center, displacement=2., rotation=True):
    """
    Generates molecule representation.
    Note, the pocket and ligand should rotate simultaneously, we thought the pocket center is the original point
    """
    pkt_sigmas, lig_sigmas = dual_sigmas
    pkt_coords, lig_coords = dual_coords
    pkt_center, lig_center = dual_center
    # Do the rotation
    if rotation:
        rrot = uniformRandomRotation()  # Rotation
        lig_coords = rotate(lig_coords, rrot, center=lig_center)
        pkt_coords = rotate(pkt_coords, rrot, center=pkt_center)

    # Note, the rondom translation is disabled, because we have a clear defination of the orginal point. 
    # Do the translation
    # center = center + (np.random.rand(3) - 0.5) * 2 * displacemen

    lig_centers2D = expanded_pcenters + pkt_center
    pkt_centers2D = expanded_pcenters + pkt_center 

    pkt_occupancy = _getOccupancyC(pkt_coords.astype(np.float32),
                               pkt_centers2D.reshape(-1, 3),
                               pkt_sigmas).reshape(size, size, size, 5)

    lig_occupancy = _getOccupancyC(lig_coords.astype(np.float32),
                               lig_centers2D.reshape(-1, 3),
                               lig_sigmas).reshape(size, size, size, 5)
    return pkt_occupancy.astype(np.float32).transpose(3, 0, 1, 2,), lig_occupancy.astype(np.float32).transpose(3, 0, 1, 2,)

def voxelize_mol(mol, rotation=True):
    '''
    Voxelize the single mol to the grid representation
    '''
    if type(mol) == Chem.rdchem.Mol:
        mol = SmallMol(mol)
    sigmas, coords, center = generate_sigmas(mol)
    if rotation:
        rrot = uniformRandomRotation() 
        coords = rotate(coords, rrot, center=center)
    
    point_centers = expanded_pcenters +center
    occupancy = _getOccupancyC(coords.astype(np.float32),
                                point_centers.reshape(-1,3),
                                sigmas).reshape(size,size,size,5)
                                
    return occupancy.astype(np.float32).transpose(3,0,1,2)

def vox_from_pair(pkt_mol, lig_mol, rotation=True):
    pkt_sigmas, pkt_coords, pkt_center = generate_sigmas(pkt_mol)
    lig_sigmas, lig_coords, lig_center = generate_sigmas(lig_mol)

    pkt_vox, lig_vox = voxelize_pkt_lig((pkt_sigmas, lig_sigmas), (pkt_coords, lig_coords), (pkt_center,lig_center), rotation=rotation)
    return pkt_vox, lig_vox


if __name__ == '__main__':
    sdf_file = './lig.sdf'
    pkt_file = './pkt.pdb'
    lig_mol = Chem.MolFromMolFile(sdf_file)
    pkt_mol = Chem.MolFromPDBFile(pkt_file)
    lig_mol, pkt_mol = align_pkt_lig_to_zero(lig_mol, pkt_mol)
    lig_mol = SmallMol(lig_mol)
    pkt_mol = SmallMol(pkt_mol)
    pkt_vox, lig_vox = vox_from_pair(pkt_mol, lig_mol)
    # 4 denotes the occup 
    np.save('pkt_occup.npy', pkt_vox[4])
    np.save('lig_occup.npy', lig_vox[4])