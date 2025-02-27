import sys
import argparse
import numpy as np
from rdkit import Chem
import pandas as pd
import torch
from models.ECloudDecipher.models.io.coati import load_e3gnn_smiles_clip_e2e
from models.ECloudDecipher.models.regression.basic_due import basic_due
from models.ECloudDecipher.data.dataset import ecloud_dataset
from models.ECloudDecipher.common.util import batch_indexable
from models.ECloudDecipher.math_tools.altair_plots import roc_plot
from models.ECloudDecipher.generative.coati_purifications import force_decode_valid_batch, purify_vector, embed_smiles
from models.ECloudDecipher.generative.embed_altair import embed_altair
from utils.chem import read_sdf, write_sdf, gen_geom_with_rdkit
import os.path as osp
from glob import glob
from tqdm import tqdm
import os
import shutil

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', choices=['cuda:0', 'cpu'], default='cuda:0',help='Device')
    arg_parser.add_argument('--input_ecloud', type=str, default = 'play_around/example/BRD4_gen_ecloud.npy')
    arg_parser.add_argument('--model', type=str, default = 'model_ckpts/ecloud_smiles_67.pkl')
    arg_parser.add_argument('--num_gen', type=int, default=100)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--noise', type=float, default=0.6)
    arg_parser.add_argument('--output', type=str, default='output/test.sdf')
    args = arg_parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file {args.model} does not exist, please check the path.")
        sys.exit(1)
    
    DEVICE = args.device
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=DEVICE,
        # model parameters to load.
        doc_url=args.model,
    )
    eclouds = torch.Tensor(np.load(args.input_ecloud)).to(torch.float).to(DEVICE).unsqueeze(0)

    save_base = osp.dirname(args.output)
    os.makedirs(save_base, exist_ok=True)
    eclouds = torch.Tensor(np.load(args.input_ecloud)).to(torch.float).to(DEVICE).unsqueeze(0)
    print('Start generating molecules...')
    mols_list = encoder.decipher_eclouds_to_mols(eclouds, tokenizer, '[SMILES]', noise_scale=args.noise, batch_repeat=args.batch_size, total_gen=args.num_gen)

    mols_list = [Chem.MolFromSmiles(s) for s in mols_list]
    mols_list = [m for m in mols_list if m is not None]
    mols_list = [gen_geom_with_rdkit(m) for m in mols_list]
    write_sdf(mols_list, args.output)
    print(f'{len(mols_list)} mols saved to {args.output}')