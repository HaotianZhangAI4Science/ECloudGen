import torch
import torch.nn as nn

from models.ECloudDiff.resnet import ResNet3D


class Ligand3DCNNEncoder(nn.Module):
    def __init__(self, cfg):
        super(Ligand3DCNNEncoder, self).__init__()
        self.cfg = cfg
        base_dim = cfg.MODEL.LIG_ENCODER.BASE_DIM
        num_block = cfg.MODEL.LIG_ENCODER.NUM_BLOCK
        output_dim = cfg.MODEL.LIG_ENCODER.OUTPUT_DIM
        factor = cfg.MODEL.LIG_ENCODER.CNN_FACTOR
        self.backbone = ResNet3D(input_dim=1,
                                 base_dim=base_dim,
                                 num_block=num_block,
                                 output_dim=output_dim,
                                 factor=factor)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

class Pocket3DCNNEncoder(nn.Module):
    def __init__(self, cfg):
        super(Pocket3DCNNEncoder, self).__init__()
        self.cfg = cfg
        base_dim = cfg.MODEL.PKT_ENCODER.BASE_DIM
        num_block = cfg.MODEL.PKT_ENCODER.NUM_BLOCK
        output_dim = cfg.MODEL.PKT_ENCODER.OUTPUT_DIM
        factor = cfg.MODEL.PKT_ENCODER.CNN_FACTOR
        self.backbone = ResNet3D(input_dim=1,
                                 base_dim=base_dim,
                                 num_block=num_block,
                                 output_dim=output_dim,
                                 factor=factor)

    def forward(self, x):
        # default x dim: [N, D, H, W, C], where C denotes channel
        if len(x.size()) == 5:
            x = x.permute(0, 4, 1, 2, 3).contiguous() 
        elif len(x.size()) == 4:
            x = x.unsqueeze(1) # [N, 1, D, H, W]
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x