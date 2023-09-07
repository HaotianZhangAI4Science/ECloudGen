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
    parser.add_argument('--ecloud_pkt', type='str',default=False)
    parser.add_argument('--save_path', type='str',default=None)
    parser.add_argument('--animation', default=False, action='store_true')
    args = parser.parse_args()
    
    ecloud_ref = np.load(args.ecloud_ref)
    ecloud_pkt = np.load(args.ecloud_pkt)
    
    if len(ecloud_ref.shape) == 4:
        ecloud_ref = ecloud_ref[0]
    if len(ecloud_pkt.shape) == 4:
        ecloud_pkt = ecloud_pkt[0]

    ecloud_ref = ecloud_ref.astype(np.float32)
    ecloud_pkt = ecloud_pkt.astype(np.float32)

    if args.save_path is None:
        args.save_path = osp.basename(args.ecloud_ref).split('.')[0] + '_pktlig'

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    y_slice_indices = np.arange(0, 64, 1)
    num_grids = ecloud_ref.shape[0]
    x_coord, y_coord, z_coord = np.meshgrid(np.arange(num_grids), np.arange(num_grids), np.arange(num_grids))
    levels1 = [0.15, 0.3 , 0.45, 0.6 , 0.75, 0.9 , 1.05, 1.2 , 1.35, 1.5 ,
        1.65, 1.8 , 1.95, 2.1 , 2.25, 2.4 , 2.55, 2.7 , 2.85, 3.  , 3.15,
        3.3 , 3.45, 3.6 , 3.75, 3.9 , 4.05, 4.2 , 4.35, 4.5 , 4.65, 4.8 ,
        4.95, 5.1 , 5.25, 5.4 ]
    
    for idx in y_slice_indices:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # def the value of each point on 2D plane
        slice_2D_z = ecloud_ref[:, :, idx]
        slice_2D_pkt_z = ecloud_pkt[:, :, idx]

        x_slice = x_coord[:, :, idx]
        y_slice = y_coord[:, :, idx]

        contourf = ax.contourf(x_slice, y_slice, slice_2D_z, levels=levels1, cmap='coolwarm')
        contour = ax.contour(x_slice, y_slice, slice_2D_z, levels= [0.025, 0.075, 0.1, 0.15, 0.2], cmap='Reds')  # 'k' for black contour lines
        contour_pkt = ax.contour(x_slice, y_slice, slice_2D_pkt_z, levels=[0., 0.07, 0.1, 0.5 , 0.75, 1], cmap='Blues')  # 'k' for black contour lines
        
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