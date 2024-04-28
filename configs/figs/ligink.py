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
        args.save_path = osp.basename(args.ecloud_ref).split('.')[0] + '_ink'
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)


    z_slice_indices = np.arange(0, 64, 1)
    num_grids = ecloud_ref.shape[0]
    x_coord, y_coord, z_coord = np.meshgrid(np.arange(num_grids), np.arange(num_grids), np.arange(num_grids))
    
    for idx in z_slice_indices:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        slice_2D_z = ecloud_ref[:, :, idx]
        x_slice = x_coord[:, :, idx]
        y_slice = y_coord[:, :, idx]
        contourf = ax.contourf(x_slice, y_slice, slice_2D_z, levels=50, cmap='Greys')
        contour = ax.contour(x_slice, y_slice, slice_2D_z, levels= [0.025, 0.075, 0.1, 0.15, 0.2], colors='k')  # 'k' for black contour lines
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
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