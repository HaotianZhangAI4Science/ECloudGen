from typing import List, Union
import torch

from torch import nn
import torch.nn.functional as F
from models.common import FFN

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


class GPT2PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix token.
    
    Input Shape: (batch_size, size, size, size)

    Output Shape: (num_layers, (2, batch_size, num_heads, prefix_length, num_embd_per_head))
    '''
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            input_size: int, # hidden size
            subencoder_type: str = 'conv3d', # ['conv3d', 'vit']
            prefix_len_ecloud: int = 128,
            prefix_len_condition: int = 5,
            prefix_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_size = input_size # GPT's embedding size
        self.n_embd_per_head = self.input_size // self.num_heads
        self.prefix_len_ecloud = prefix_len_ecloud
        self.prefix_len_condition = prefix_len_condition
        self.prefix_dropout = prefix_dropout
        self.condition_len = 3 # hbd, hba, tpsa, mw

        if subencoder_type == 'conv3d':
            self.subencoder = Conv3DEncoder(d_model=self.input_size)
            self.encoded_sl = 64 # 512 for 64^3
        else:
            # 'vit'
            self.subencoder = ShapePretrainingEncoder(patch_size=8)
            self.encoded_sl = 343

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.len_trans = FFN(self.encoded_sl, self.prefix_len_ecloud, self.prefix_len_ecloud)

        # transform (sl, bz, d_model) to (sl, bz, 2 * num_layers * num_heads * n_embd_per_head)
        self.control_trans = FFN(self.input_size, 2 * self.num_layers * self.input_size, 2 * self.num_layers * self.input_size)
        
        # condition size: (bz, condition_len) here condition len is 4(hbd, hba, tpsa, mw)
        # (bz, condition_len) to (bz, 2 * prefix_len_ecloud * num_layers )
        # bz, 2 * 5 * 12 * 768
        # self.condition_trans = FFN(self.condition_len, 2 * self.prefix_len_condition * self.num_layers * self.input_size, 2 * self.prefix_len_condition * self.num_layers * self.input_size)
        # 
        self.condition_trans = FFN(self.condition_len, 2 * self.prefix_len_condition * self.num_layers, 2 * self.prefix_len_condition * self.num_layers * self.input_size)
        

    def forward(self, ecloud: torch.Tensor, condition: torch.Tensor):
        input_tokens = self.subencoder(ecloud) # (sl, bz, d_model)
        # (bz, d_model, sl)
        input_tokens = input_tokens.transpose(0, 1).transpose(1, 2)

        input_tokens = self.len_trans(input_tokens) # (bz, d_model, prefix_len)

        # (bz, d_model, prefix_len) -> (bz, prefix_len, d_model)
        input_tokens = input_tokens.transpose(1, 2)

        past_key_values_ecloud = self.control_trans(input_tokens) # (prefix_len, bz, 2 * num_layers * num_heads * n_embd_per_head)
        
        past_key_values_conditions = self.condition_trans(condition) # (bz, 2 * prefix_len_ecloud * num_layers )
        # print(past_key_values_conditions.shape)

        curr_batch_size = ecloud.shape[0]

        # choice No.1: Each layer has its own prefix
        past_key_values_ecloud = past_key_values_ecloud.view(
            curr_batch_size, self.prefix_len_ecloud , 2 * self.num_layers, self.num_heads, self.n_embd_per_head
        )

        past_key_values_conditions = past_key_values_conditions.view(
            curr_batch_size, self.prefix_len_condition, 2 * self.num_layers, self.num_heads, self.n_embd_per_head
        )
        # concat two prefix
        past_key_values = torch.cat([past_key_values_ecloud, past_key_values_conditions], dim=1) # (bz, prefix_len, 2 * num_layers, num_heads, n_embd_per_head)

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute(2, 0, 3, 1, 4).split(2) # 2*num_layers, bs, num_heads, prefix_len, n_embd_per_head
        # after split: 
        return past_key_values
