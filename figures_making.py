import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Figure Making")
    parser.add_argument('--w_pkt', default=False, action='store_true')
    args = parser.parse_args()

    ecloud_pre = np.load('lig_ecloud.npy')
    pkt_ecloud = np.load('pkt_ecloud.npy')
    ecloud_ref = np.load('lig_ref.npy')

    for k in tqdm(range(ecloud_pre.shape[0])):
        # _ref = pkt_ecloud[0] + ecloud_pre[k]
        _ref = ecloud_pre[k]
        size = _ref.shape[0]
        data = _ref.reshape(-1, 1)
        # 创建坐标轴
        x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))

        # 绘制3D热度图
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=data.flatten(),
            isomin=np.min(data),
            isomax=np.max(data),
            opacity=0.085,
            surface_count=17, 
            ))

        pio.write_image(fig, f'./samples/{k}.png')

    if args.w_pkt:
        for k in tqdm(range(ecloud_pre.shape[0])):
            _ref = pkt_ecloud[0] + ecloud_pre[k]
            # _ref = ecloud_pre[k]
            size = _ref.shape[0]
            data = _ref.reshape(-1, 1)
            # 创建坐标轴
            x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))

            # 绘制3D热度图
            fig = go.Figure(data=go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=data.flatten(),
                isomin=np.min(data),
                isomax=np.max(data),
                opacity=0.085,
                surface_count=17, 
                ))

            pio.write_image(fig, f'./samples/{k}_wpkt.png')

    for k in tqdm(range(ecloud_ref.shape[0])):
        _ref = ecloud_ref[0]
        # _ref = ecloud_pre[k]
        size = _ref.shape[0]
        data = _ref.reshape(-1, 1)
        # 创建坐标轴
        x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))

        # 绘制3D热度图
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=data.flatten(),
            isomin=np.min(data),
            isomax=np.max(data),
            opacity=0.085,
            surface_count=17, 
            ))

        pio.write_image(fig, f'./samples/{k}_ori.png')