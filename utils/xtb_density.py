"""
Calculate charge density using xTB, only works for GNU/Linux.
"""

import os
import uuid
import numpy
import shutil
import rdkit as rd
import rdkit.Chem
import cubtools


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
    def __init__(self, xtb_command: str = 'xtb') -> None:
        self.clean()
        self.rootdir = os.getcwd()
        self.workdir = os.getcwd() + '/' + uuid.uuid4().hex
        os.mkdir(self.workdir)
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
                meta.pop('atoms')
                result = {'density': density * (BOHR ** 3), 'meta': meta}
            except:
                result = {}
        else:
            result = {}
        os.chdir(self.rootdir)
        return result

    @property
    def err_msg(self):
        if (self._errno == 0):
            return ''
        else:
            return open(self.workdir + 'log', 'r').read()
