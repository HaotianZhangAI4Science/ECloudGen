"""
Calculate charge density using xTB, only works for GNU/Linux.
"""

import os
import uuid
import numpy
import shutil
import rdkit as rd
import rdkit.Chem
try:
    import cubtools
    from cubtools import write_cube
except: 
    import utils.cubtools as cubtools
    from utils.cubtools import write_cube

import numpy as np


BOHR = 1.8897259886
XTBINP = r"""
$cube
    step={:f}
$end
$write
    density=true
    spin density=false
    fod=false
    charges=false
    mulliken=false
    geosum=false
    inertia=false
    mos=false
    wiberg=false
$end
""".format(0.5 * BOHR)


class CDCalculator(object):
    '''
    simple wrapper for xTB to calculate charge density
    Constructor:
        xtb_command: the command to run xTB, default is 'xtb', or you can use the binary installation path
    Functions:
        calculate(mol): calculate the charge density of a molecule
        clean(): clean the temporary files
    '''

    def __init__(self, xtb_command: str = 'xtb') -> None:
        self.clean()
        self.rootdir = os.getcwd()
        self.workdir = os.getcwd() + '/temp/' + uuid.uuid4().hex
        os.makedirs(self.workdir)
        self.xtb_command = xtb_command
        os.environ['KMP_STACKSIZE'] = '1G'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        with open(self.workdir + '/xtb.inp', 'w') as fp:
            print(XTBINP, file=fp, flush=True)
        self._errno = 0

    def __del__(self):
        self.clean()

    def clean(self):
        try:
            shutil.rmtree(self.workdir)
        except:
            pass

    def calculate(self, mol: rd.Chem.rdchem.Mol) -> dict:
        os.chdir(self.workdir)
        rd.Chem.rdmolfiles.MolToXYZFile(mol, 'tmp.xyz')
        command = self.xtb_command + ' --norestart --input xtb.inp tmp.xyz' + \
                                     ' 2> ' + os.devnull + ' > log'
        self._errno = os.system(command)
        if (self._errno == 0):
            try:
                density, meta = cubtools.read_cube('density.cub')
                meta['org'] = numpy.array(meta['org']) / BOHR
                meta['len'] = (numpy.array(meta['xvec']) * meta['nx'] +
                               numpy.array(meta['yvec']) * meta['ny'] +
                               numpy.array(meta['zvec']) * meta['nz']) / BOHR
                # meta.pop('atoms')
                result = {'density': density * (BOHR ** 3), 'meta': meta}
            except Exception as e:
                print(e)
                result = {}
        else:
            print(command)
            print('xtb has failed')
            result = {}
        os.chdir(self.rootdir)
        return result

    @property
    def err_msg(self):
        if (self._errno == 0):
            return ''
        else:
            return open(self.workdir + 'log', 'r').read()

from scipy.interpolate import RegularGridInterpolator

def ecloud2grid(cub):
    """
    compute the grid coordinates of the ecloud dict
    Parameters:
        cub: ecloud dict
    """

    x_len, y_len, z_len = cub['meta']['len'] # three dimensional length of the box
    x_llc, y_llc, z_llc = cub['meta']['org']
    x_cell, y_cell, z_cell = cub['meta']['nx'], cub['meta']['ny'], cub['meta']['nz']


    X, Y, Z = np.meshgrid(np.arange(float(x_cell)), np.arange(float(y_cell)), np.arange(float(z_cell)))

    X *= x_len / x_cell 
    Y *= y_len / y_cell  
    Z *= z_len / z_cell 
    # grid_coords = np.stack([X, Y, Z], axis=-1)
    return X+x_llc, Y+y_llc, Z+z_llc

def interplot_ecloud(ecloud, new_gridcoord):
    """ 
    Assign the density of the ecloud to the new coordinates
    Parameters:
        ecoud: ecloud dict
        new_gridcoord: new coordinates of the ecloud
    """
    X, Y, Z = ecloud2grid(ecloud)
    X_1d = X[0, :, 0]
    Y_1d = Y[:, 0, 0]
    Z_1d = Z[0, 0, :]
    
    interpolator = RegularGridInterpolator((X_1d, Y_1d, Z_1d), ecloud['density'], 
                                        bounds_error=False, fill_value=0.0)
    new_X,new_Y, new_Z = new_gridcoord
    new_points = np.array([new_X.flatten(), new_Y.flatten(), new_Z.flatten()]).T
    new_density = interpolator(new_points)

    return new_density

def write_new_cube(new_gridcoord, new_density, meta, fname):
    """
    Write the new cube file
    Parameters:
        new_gridcoord: new coordinates of the ecloud (Unit: A) shape = (3, nx, ny, nz)
        new_density: new density of the ecloud
        meta: meta data of the ecloud
        fname: name of the new cube file
    """
    new_X,new_Y, new_Z = new_gridcoord
    vec_x = new_X[1, 0, 0] - new_X[0, 0, 0]
    vec_y = new_Y[0, 1, 0] - new_Y[0, 0, 0]
    vec_z = new_Z[0, 0, 1] - new_Z[0, 0, 0]

    new_meta = {'org':np.array([np.min(new_X), np.min(new_Y), np.min(new_X)])*BOHR, #llc
            'xvec': np.array([vec_x, 0, 0])*BOHR,
            'yvec': np.array([0, vec_y, 0])*BOHR,
            'zvec': np.array([0, 0, vec_z])*BOHR,
            'nx': new_gridcoord[0].shape[0],
            'ny': new_gridcoord[0].shape[1],
            'nz': new_gridcoord[0].shape[2],
            'atoms': meta['atoms'],
    }

    new_density = new_density.reshape(new_X.shape) 
    write_cube(new_density, new_meta, fname)

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

step = 0.5
size = 24
cell_number = [size, size, size]
llc = (np.zeros(3) - float(size * step / 2))
# Now, the box is 24×24×24 A^3

expanded_pcenters = BuildGridCenters(llc, cell_number, step)
# Now, the expanded_pcenters is the coordinates of the grid points


if __name__ == '__main__':
    # reconstruct cube file
    from rdkit import Chem
    from cubtools import write_cube
    import numpy as np

    claculator = CDCalculator(xtb_command='xtb')
    mol = Chem.MolFromMolFile('./90.sdf')
    ecloud = claculator.calculate(mol)
    recon_meta = {'org': ecloud['meta']['org']*BOHR,
                'xvec': np.array(ecloud['meta']['xvec']),
                'yvec': np.array(ecloud['meta']['yvec']),
                'zvec': np.array(ecloud['meta']['zvec']),
                'nx': ecloud['meta']['nx'],
                'ny': ecloud['meta']['ny'],
                'nz': ecloud['meta']['nz'],
                'len': ecloud['meta']['len']*BOHR,
                'atoms': ecloud['meta']['atoms']}

    write_cube(ecloud['density']/(BOHR**3), recon_meta, './recon.cub')
    # recon_x, recon_y, recon_z = cub2grid(result)
    # np.sum(ecloud['density'] != result['density'])
    # TEST: Interpolate the ecloud to the new coordinates
    new_density = interplot_ecloud(ecloud, expanded_pcenters.transpose(3, 0, 1, 2))
    write_new_cube(expanded_pcenters.transpose(3,0,1,2), new_density/ (BOHR**3), ecloud['meta'], './new.cub')


