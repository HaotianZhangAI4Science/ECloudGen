
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from functools import partial
from pytorch_lightning import LightningDataModule
from tokenizers.implementations import BaseTokenizer
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm


class DecipherDataset(Dataset):
    """
    Data contains the ecloud, conditions, and following smiles
    """

    def __init__(self, data_root:str, tokenizer: BaseTokenizer, max_length: int=64, csv_file='./mol_data/moses2.csv', tokenizer_filename='./vocab/tokenizer.json') -> None:
        super().__init__()
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length - 1 # save space for prefix tokens
        self.csv_file = csv_file
        self.tokenizer_filename = tokenizer_filename

        if data_root.endswith('.h5'):
            self.h5_path = data_root
            self.data_root = data_root[:-3]
        else:
            self.h5_path = data_root + '.h5'

    def __len__(self) -> int:
        f = h5py.File(self.h5_path, 'r')
        # print(len(f['eclouds']))
        return len(f['eclouds'])
    
    def __getitem__(self, i: int):

        if not os.path.exists(self.h5_path):
            self.create_h5()

        f = h5py.File(self.h5_path, 'r')
        ecloud_item = f['eclouds'][i]
        condition_item = f['conditions'][i]
        smiles_item = f['smiles'][i]
        f.close()
        return smiles_item, ecloud_item, condition_item
  
    
from vocab.tokenization import SMILESBPETokenizer
def create_h5(h5_path, ecloud_path, mol_csv, max_length=64, tokenizer_filename='./vocab/tokenizer.json'):
    save_file = h5_path
    files = os.listdir(ecloud_path)
    num_files = len(files)
    mols_data = pd.read_csv(mol_csv)
    max_length = max_length
    tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
        tokenizer_file=tokenizer_filename,
        model_max_length=max_length
    )
    with h5py.File(save_file, 'w') as hf:
        hf.create_dataset(
            name='eclouds',
            shape=(num_files, 32, 32, 32),
            dtype=np.float16)

        hf.create_dataset(
            name='smiles',
            shape=(num_files,max_length),
            dtype=np.int16)

        hf.create_dataset(
            name='conditions',
            shape=(num_files, 3),
            dtype=np.float16)

        for i in tqdm(range(num_files)):
            file_name = os.path.join(ecloud_path, files[i])
            mol_id = int(files[i][:-4])
            ecloud = np.load(file_name)
            data = mols_data.iloc[mol_id]
            smiles = data.SMILES
            smiles = np.array(tokenizer.encode(smiles,truncation=True,padding='max_length',max_length=max_length))
            print(smiles.shape)
            condition = [data.qed, data.logp, data.TPSA]
            try:
                hf['smiles'][i] = smiles
                hf['eclouds'][i] = ecloud
                hf['conditions'][i] = condition
            except:
                print('Error in: ', i)

    print('Creation Done.')

class PrefixTuningDataModule(LightningDataModule):
    """
    Lightning data module for autoregressive language modeling prefix tuning.
    """


    def __init__(self, 
                 data_root: str, 
                 tokenizer: BaseTokenizer, 
                 batch_size: int = 128, 
                 num_workers: int = 0, 
                 max_length: int = 64) -> None:
        super().__init__()
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = DecipherDataset(self.data_root, self.tokenizer, self.max_length)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader],
                                        Dict[str, DataLoader]]:
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.num_workers)

def collate_fn(examples):
    batch = {
        "input_ids": torch.stack([torch.LongTensor(e[0]) for e in examples], dim=0),
        "prefix_tokens_ecloud": torch.stack([torch.FloatTensor(e[1]) for e in examples], dim=0),
        "prefix_tokens_condition": torch.stack([torch.FloatTensor(e[2]) for e in examples], dim=0)
    }
    labels = batch["input_ids"].clone()
    labels[labels == 0] = -100 # ignore loss for padding tokens
    batch["labels"] = labels
    return batch