import numpy as np
import argparse
import shutil
import torch
import os
import argparse
import json
from utils.chem import pocket_trunction, read_sdf, read_pkl
from models.ECloudDiff.latentdiff import EcloudLatentDiffusionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import pickle
from utils.protein_ligand import protocol, get_occu_ecloud_pair
from glob import glob
from rdkit import Chem
from omegaconf import OmegaConf
from typing import List, Tuple


def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(os.getcwd(), fn))

class ReverseSampler():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.load_model(self.cfg.checkpoint)
        self.model.eval()
        self.model = self.model.cuda()

    def load_model(self, ckpt_path):

        model = EcloudLatentDiffusionModel(self.cfg)
        assert os.path.exists(ckpt_path), "please input a ckpt path and ensure it right ! "
        pretrained_dict = torch.load(self.cfg.checkpoint, map_location='cpu')

        model_dict = model.state_dict()
        if 'model' in pretrained_dict:
            pretrained_dict = {k: v for k, v in pretrained_dict['model'].items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}

        total = len(model_dict.keys())
        rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
        print('Parameter Loading:', rate, '...')
        print([k for k in pretrained_dict.keys() if k not in model_dict])
        print([k for k in model_dict.keys() if k not in pretrained_dict])

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"  =====================================================================================")
        print(f"  Load pretrained model from {self.cfg.checkpoint}")
        print(f"  =====================================================================================")
        return model


    def sample_from_pdb(self, pdb_file, lig_file, num_gen=8, batch=1, protein_channel_full=False):
        
        lig_mol = read_sdf(lig_file)[0]
        centroid = lig_mol.GetConformer().GetPositions().mean(axis=0)
        pkt_file = pocket_trunction(pdb_file, centroid=centroid)
        pkt_mol = Chem.MolFromPDBFile(pkt_file)
        os.remove(pkt_file)

        pkt_channel, lig_density = get_occu_ecloud_pair(pkt_mol, lig_mol, grid_protocol=protocol(32))
        accelerator = Accelerator(mixed_precision='fp16')

        self.model = accelerator.prepare(self.model)

        device = next(self.model.parameters()).device
        if protein_channel_full:
            pkt_ecloud = torch.tensor(pkt_channel, dtype=torch.float16).to(device).permute(3,0,1,2)
        else:
            pkt_ecloud = torch.tensor(pkt_channel[:,:,:,4], dtype=torch.float16).to(device).unsqueeze(0)
        print(pkt_ecloud.shape)
        ecloud_ref = torch.tensor(lig_density, dtype=torch.float16).to(device).unsqueeze(0)
        ecloud_pre_list = []
        
        sample_shape = tuple([batch] + self.cfg.sample_shape)
        count = 0
    
        pkt_z = self.model.PocketEncoder(pkt_ecloud)
        pkt_z = pkt_z.permute(0, 4, 1, 2, 3).contiguous()
        while count < num_gen:
            count += batch
            lig_z = self.model.p_sample_loop(
                self.model,
                sample_shape,
                condition=pkt_z.repeat(batch, 1, 1, 1, 1),
                clip_denoised=False,
                progress=True,
            )
            z = torch.cat([pkt_z.repeat(batch, 1, 1, 1, 1), lig_z], dim=1)
            z = z.permute(0, 2, 3, 4, 1).contiguous()
            logits = self.model.EcloudDecoder(z)
            ecloud_pre_list.append(logits.squeeze(1))
        ecloud_pre = torch.cat(ecloud_pre_list).cpu().detach().numpy().astype(np.float16)
        ecloud_ref = ecloud_ref.cpu().detach().numpy().astype(np.float16)
        pkt_ecloud = pkt_ecloud.cpu().detach().numpy().astype(np.float16)
        return ecloud_pre, pkt_ecloud, ecloud_ref


def updata_cfg(cfg, args):
    for k, v in vars(args).items():
        cfg[k] = v
    return cfg

if __name__ == '__main__':
    # conda activate ecloud
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default='./example')
    parser.add_argument('--config', type=str, default='./configs/eclouddiff.yml')
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./ckpt/32_ldm_4channel_ddpm_100.pt')
    parser.add_argument('--coords_method', type=str, default='sgd', choices=['eig', 'sgd'])
    parser.add_argument('--num_gen', type=int, default=8, help=" The number of generation sample ")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--sample_shape', type=List[int], default=(256, 16, 16, 16), help="the number of atoms")
    parser.add_argument('--pdb_file', type=str, default='./play_around/peptide_example/7ux5_protein.pdb', help="")
    parser.add_argument('--lig_file', type=str, default='./play_around/peptide_example/7ux5_peptide.sdf', help="")
    args = parser.parse_args()

    cfg = OmegaConf.load(get_abs_path(args.config))
    cfg = updata_cfg(cfg, args)

    cfg.MODEL.PKT_ENCODER.CHANNELS = 1

    sampler = ReverseSampler(cfg)
    cfg.sample_shape = [256, 8, 8, 8]

    ecloud_pre, pkt_ecloud, ecloud_ref = sampler.sample_from_pdb(args.pdb_file, args.lig_file, num_gen=args.num_gen, batch=args.batch, protein_channel_full=False)


    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    lig_filename = os.path.basename(args.lig_file)[:-4]
    pdb_filename = os.path.basename(args.pdb_file)[:-4]
    np.save(os.path.join(args.outputs_dir, f'./{lig_filename}_ecouldgen.npy'), ecloud_pre)
    np.save(os.path.join(args.outputs_dir, f'./{pdb_filename}_ecloud.npy'), pkt_ecloud)
    np.save(os.path.join(args.outputs_dir, f'./{lig_filename}_ecloudref.npy'), ecloud_ref)
    shutil.copy(args.pdb_file, args.outputs_dir)
    shutil.copy(args.lig_file, args.outputs_dir)
    print(f"Save the results to {args.outputs_dir}")


