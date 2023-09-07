import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
from glob import glob 
import os.path as osp
import argparse
import os
import imageio

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Figure Making")
    parser.add_argument('--ecloud_ref', type='str',default=False)
    parser.add_argument('--save_path', type='str',default=None)
    parser.add_argument('--animation', default=False, action='store_true')
    args = parser.parse_args()
    
    ecloud_ref = np.load(args.ecloud_ref)
    
    if len(ecloud_ref.shape) == 4:
        ecloud_ref = ecloud_ref[0]
    ecloud_ref = ecloud_ref.astype(np.float32)

    if args.save_path is None:
        args.save_path = osp.basename(args.ecloud_ref).split('.')[0] + '_lava'

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    y_slice_indices = np.arange(0, 64, 1)
    num_grids = ecloud_ref.shape[0]
    x_coord, y_coord, z_coord = np.meshgrid(np.arange(num_grids), np.arange(num_grids), np.arange(num_grids))
    
    for idx in y_slice_indices:
        slice_2D_y = ecloud_ref[:, idx, :]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(slice_2D_y, cmap='magma', origin='lower')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(f'{args.save_path}/{idx}.png', dpi=300)

    if args.animation:
        filenames = glob(f'./{args.save_path}/*')
        filenames = sorted(filenames, key=lambda x: int(x.split('/')[-1].split('.')[0]))    
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))

        default_file_name = osp.basename(ecloud_ref).split('.')[0]
        imageio.mimsave(f'{args.save_path}/{default_file_name}.gif', images, duration=0.1)