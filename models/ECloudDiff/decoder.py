import torch
import torch.nn as nn

class Ecloud3DCNNDecoder(nn.Module):
    def __init__(self, cfg):
        super(Ecloud3DCNNDecoder, self).__init__()
        self.cfg = cfg
        dim = cfg.MODEL.LIG_ENCODER.BASE_DIM * (cfg.MODEL.LIG_ENCODER.NUM_BLOCK ** 2)
        input_dim = cfg.MODEL.LIG_ENCODER.OUTPUT_DIM + cfg.MODEL.PKT_ENCODER.OUTPUT_DIM
        self.proj_conv = nn.Conv3d(input_dim, dim, kernel_size=(1, 1, 1), stride=1)
        self.upsample_block = nn.ModuleList()
        for k in range(cfg.MODEL.DECODER.NUM_BLOCK):
            self.upsample_block.append(
                nn.Sequential(
                    nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.BatchNorm3d(dim // 2),
                    nn.ReLU(),
                )
            )
            dim //= 2
        self.head = nn.Conv3d(dim, 1, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.proj_conv(x)
        for blk in self.upsample_block:
            x = blk(x)
        x = self.head(x)
        return x
