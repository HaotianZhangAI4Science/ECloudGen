"""
loads data used for training COATI.

c.f. make_cache. which does a lot of aggs. 
"""
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.datapipes.iter import Shuffler
import h5py




class ecloud_dataset(Dataset):
    def __init__(
        self,
        data_dir = 'ecloud_coati.h5'
    ):
        super().__init__()
        self.data_dir = data_dir
        if self.data_dir.endswith('.pt'):
            self.f = torch.load(self.data_dir)
        else:
            self.f = h5py.File(self.data_dir, 'r')
        self.summary = {"dataset_type": "ecloud", 
                        "fields": ['eclouds', 'raw_tokens', 'augmented_tokens']}

    def __len__(self):
        return len(self.f['eclouds']) 
    
    def __getitem__(self, i):
        eclouds = self.f['eclouds'][i]
        raw_tokens = self.f['raw_tokens'][i]
        augmented_tokens = self.f['augmented_tokens'][i]
        return {'eclouds': eclouds, 'raw_tokens': raw_tokens, \
                'augmented_tokens': augmented_tokens}


