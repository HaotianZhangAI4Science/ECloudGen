import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=0, stride=1):
        super(Conv3d, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.spatial_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.temporal_conv = nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c d h w -> (b d) c h w', b=B, d=D)
        x = self.spatial_conv(x)

        C, H, W = x.size()[-3:]
        x = rearrange(x, '(b d) c h w -> (b h w) c d', b=B, d=D, h=H, w=W)
        x = self.temporal_conv(x)

        D = x.size(-1)
        x = rearrange(x, '(b h w) c d -> b c d h w', b=B, d=D, h=H, w=W)
        return x


mode = '3d'
if mode == '(2+1)d':
    Conv3D = Conv3d
else:
    Conv3D = nn.Conv3d


class ResNet3DBlock(nn.Module):
    def __init__(self, dim=256, factor=2, down_sample=False):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = Conv3D(dim, dim//factor, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(dim//factor)
        self.act1 = nn.ReLU()
        self.conv2 = Conv3D(dim//factor, dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(dim)
        self.act2 = nn.ReLU()
        self.down_sample= down_sample
        if self.down_sample:
            self.conv3 = Conv3D(dim, dim*2, kernel_size=1, stride=1)
        else:
            self.conv3 = Conv3D(dim, dim, kernel_size=1, stride=1)

    def forward(self, x):
        assert len(x.shape) == 5
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = res + x
        x = self.conv3(x)
        if self.down_sample:
            x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        return x

class ResNet3D(nn.Module):
    def __init__(self, input_dim, base_dim, num_block, output_dim, factor):
        super(ResNet3D, self).__init__()
        self.conv = nn.Conv3d(input_dim, base_dim, kernel_size=1, stride=1)
        self.block = nn.ModuleList()
        for i in range(num_block):
            # 每个block都下采样
            self.block.append(
                ResNet3DBlock(dim=base_dim, factor=factor, down_sample=True)
            )
            base_dim *= 2
        # input_dim = base_dim * 2^num_block  # 128 -> 512, 512 -> 256
        self.output = Conv3D(base_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.type(self.conv.weight.dtype)
        x = self.conv(x)
        for blk in self.block:
            x = blk(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    net = ResNet3D(input_dim=8, base_dim=128, num_block=2, output_dim=256, factor=2)
    print(net)
    x = torch.rand(4, 8, 32, 32, 32)
    y = net(x)
    print(y.shape)