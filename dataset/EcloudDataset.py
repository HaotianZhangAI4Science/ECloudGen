import os
import random
import numpy as np
import h5py
import torch
import logging
from datasets import register_datasets
from ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from ecloud_utils.grid import rotate, BuildGridCenters, generate_sigmas
from moleculekit.util import uniformRandomRotation
from utils.chem import *
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import _getOccupancyC
logging.getLogger("moleculekit").setLevel(logging.WARNING)

# calculater = CDCalculator(xtb_command='/mnt/e/tangui/ECloudGen_demo/xtb-bleed/bin/xtb')
@register_datasets(['ecloud'])
class ProteinLigandPairDataset():
    def __init__(self, cfg, task, mode='train'):
        self.cfg = cfg
        data_path = self.cfg.DATA.DATA_ROOT
        if mode == 'train':
            self.file_path = os.path.join(data_path, self.cfg.DATA.TRAIN_DATA)
        elif mode == 'valid':
            self.file_path = os.path.join(data_path, self.cfg.DATA.VALID_DATA)
        elif mode == 'test':
            self.file_path = os.path.join(data_path, self.cfg.DATA.TEST_DATA)

        with h5py.File(self.file_path, 'r') as h:
            self.num_data = h['ecloud_pocket'].shape[0]

    @classmethod
    def build_datasets(cls, cfg, task, mode):
        return cls(cfg, task, mode)

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        with h5py.File(self.file_path, 'r') as hf:
            pocket_ecloud = hf['ecloud_pocket'][item]
            ligand_ecloud = hf['ecloud_ligand'][item]
        random_id = int(np.random.choice(5, 1)[0])
        pocket_ecloud = pocket_ecloud[random_id]
        ligand_ecloud = ligand_ecloud[random_id]
        return {'pkt': pocket_ecloud, 'lig': ligand_ecloud}

    def collator(self, samples):
        batch_pkt_tensor = []
        batch_lig_tensor = []
        for sample in samples:
            batch_pkt_tensor.append(torch.tensor(sample['pkt']).unsqueeze(0))
            batch_lig_tensor.append(torch.tensor(sample['lig']).unsqueeze(0))
        batch_pkt_tensor = torch.cat(batch_pkt_tensor)
        batch_lig_tensor = torch.cat(batch_lig_tensor)
        batch = {
            'net_input': {
                'pkt': batch_pkt_tensor,
                'lig': batch_lig_tensor,
            },
            'net_output': None
        }
        return batch


# @register_datasets(['ecloud'])
class ProteinLigandPairDataset():
    def __init__(self, cfg, task, mode='train'):
        self.cfg = cfg
        self.grid_resolution = cfg.DATA.grid_resolution
        self.max_dist = cfg.DATA.max_dist
        self.num_atomic = cfg.DATA.num_atomic
        data_path = self.cfg.DATA.DATA_ROOT
        pkt_ids = os.listdir(data_path)
        all_data = []
        for pkt_id in pkt_ids:
            pair_path = os.path.join(data_path, pkt_id)
            if not os.path.isdir(pair_path):
                continue
            files = os.listdir(pair_path)
            for file in files:
                if file.endswith('.sdf'):
                    sdf_path = os.path.join(pair_path, file)
                    pdb_path = os.path.join(pair_path, file.split('.')[0] + '_pocket10.pdb')
                    all_data.append((sdf_path, pdb_path))
        total = len(all_data)
        if mode == 'train':
            self.data = all_data[:int(0.9*total)]
        else:
            self.data = all_data[int(0.9*total):]
        self.num_data = len(self.data)

    @classmethod
    def build_datasets(cls, cfg, task, mode):
        return cls(cfg, task, mode)

    def __len__(self):
        return self.num_data

    def build_box(self):
        boundary = 2 * self.max_dist
        resolution = self.grid_resolution  # boundy / size
        self.size = int(boundary / resolution)
        assert self.size % 1 == 0, print('Error: size must be an integer')
        self.size += 1
        N = [self.size, self.size, self.size]
        llc = (np.zeros(3) - float(self.size * resolution / 2)) + resolution / 2
        expanded_pcenters = BuildGridCenters(llc, N, resolution)
        return expanded_pcenters

    def get_ecloud(self, pkt_mol, lig_mol, expanded_pcenters):
        calculater = CDCalculator(xtb_command='xtb')
        pkt_smallmol = SmallMol(pkt_mol)

        lig_coords = lig_mol.GetConformer().GetPositions()
        lig_center = lig_coords.mean(axis=0)

        # define the pkt channel
        pkt_sigmas, pkt_coords, pkt_center = generate_sigmas(pkt_smallmol)

        # use the pkt_center as the whole center
        pkt_grids = expanded_pcenters + pkt_center
        lig_grids = expanded_pcenters + pkt_center

        # Do the rotation
        rrot = uniformRandomRotation()  # Rotation
        lig_coords = rotate(lig_coords, rrot, center=pkt_center)
        pkt_coords_ = rotate(pkt_mol.GetConformer().GetPositions(), rrot, center=pkt_center)
        pkt_coords = rotate(pkt_coords, rrot, center=pkt_center)

        pkt_channel = _getOccupancyC(pkt_coords.astype(np.float32),
                                     pkt_grids.reshape(-1, 3),
                                     pkt_sigmas).reshape(self.size, self.size, self.size, 8)

        rotated_lig_mol = set_mol_position(lig_mol, lig_coords)
        lig_ecloud = calculater.calculate(rotated_lig_mol)
        lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape([self.size, self.size, self.size])

        rotated_pkt_mol = set_mol_position(pkt_mol, pkt_coords_)
        calculater.clean()
        return pkt_channel, lig_density

    def __getitem__(self, item):
        try:
            sdf_path, pdb_path = self.data[item]
            pkt_mol = Chem.MolFromPDBFile(pdb_path)
            lig_mol = read_sdf(sdf_path)[0]
            expanded_pcenters = self.build_box()
            # print(self.data[item])
            pkt_channel, lig_density = self.get_ecloud(pkt_mol, lig_mol, expanded_pcenters)
        except:
            sdf_path, pdb_path = self.data[0]
            pkt_mol = Chem.MolFromPDBFile(pdb_path)
            lig_mol = read_sdf(sdf_path)[0]
            expanded_pcenters = self.build_box()
            pkt_channel, lig_density = self.get_ecloud(pkt_mol, lig_mol, expanded_pcenters)
        return {'pkt': pkt_channel, 'lig': lig_density}

    def collator(self, samples):
        batch_pkt_tensor = []
        batch_lig_tensor = []
        for sample in samples:
            batch_pkt_tensor.append(torch.tensor(sample['pkt']).unsqueeze(0))
            batch_lig_tensor.append(torch.tensor(sample['lig']).unsqueeze(0))
        batch_pkt_tensor = torch.cat(batch_pkt_tensor)
        batch_lig_tensor = torch.cat(batch_lig_tensor)
        batch = {
            'net_input': {
                'pkt': batch_pkt_tensor,
                'lig': batch_lig_tensor,
            },
            'net_output': None
        }
        return batch