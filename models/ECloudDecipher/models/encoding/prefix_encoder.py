from typing import List, Union
import torch
from torch import nn
import torch.nn.functional as F

class Conv3DEncoder(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, d_model=256):
        super(Conv3DEncoder, self).__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv3d(in_channels, d_model // 4, kernel_size, stride, padding, dilation)
        self.conv2 = nn.Conv3d(d_model // 4, d_model // 2, kernel_size, stride, padding, dilation)
        self.conv3 = nn.Conv3d(d_model// 2, d_model, kernel_size, stride, padding, dilation)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        bz = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(bz, -1, self.d_model)
        sl = x.size(1)
        # prepare input for decoder
        x = x.transpose(0, 1)
        

        src_padding_mask = torch.zeros((bz, sl), dtype=torch.bool).to(x.device)

        # encoder_out = x, src_padding_mask
        # return encoder_out
        return x

class Conv3DDecoder(nn.Module):
    def __init__(self, in_channels=768, kernel_size=3, stride=1, padding=1, dilation=1, d_model=768):
        super(Conv3DDecoder, self).__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv3d(in_channels, d_model // 2, kernel_size, stride, padding, dilation)
        self.conv2 = nn.Conv3d(d_model // 2, d_model // 4, kernel_size, stride, padding, dilation)
        self.conv3 = nn.Conv3d(d_model// 4, 1, kernel_size, stride, padding, dilation)
        self.pool = nn.Upsample(scale_factor=2)
        self.relu = nn.LeakyReLU()

    def forward(self, x): # (b, 768, 64)
        bz = x.size(0)
        x = x.view(bz, -1, 4, 4, 4) #(b, 768, 4, 4, 4)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x) #(b, 1, 32, 32, 32)
        x = x.squeeze(1)

        return x
