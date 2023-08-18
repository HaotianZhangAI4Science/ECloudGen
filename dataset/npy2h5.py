import os
from tqdm import tqdm
import numpy as np
import argparse
import h5py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecloud_root', type=str, default='/mnt/d/KunLai/LDM3D/ECloud_data/Ecloud/')
    parser.add_argument('--save_file', type=str, default='/mnt/d/KunLai/LDM3D/ECloud_data/ecloud_moses.h5')
    parser.add_argument('--num_files', type=int, default=1000)
    args = parser.parse_args()
    return args


def process_ecloud(ecloud_root, save_file):
    files = os.listdir(ecloud_root)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    num_files = len(files)
    smiles = []
    with h5py.File(save_file, 'w') as hf:

        hf.create_dataset(
            name='eclouds',
            shape=(num_files, 64, 64, 64),
            dtype=np.float16)

        for i in tqdm(range(num_files)):
            file_name = os.path.join(ecloud_root, str(i) + '.npy')
            ecloud = np.load(file_name)
            hf['eclouds'][i] = ecloud
            

if __name__ == "__main__":
    args = parse_args()
    ecloud_root = args.ecloud_root
    save_file = args.save_file
    num_files = args.num_files
    process_ecloud(ecloud_root, save_file)