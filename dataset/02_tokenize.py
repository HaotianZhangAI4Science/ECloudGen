import sys
sys.path.append('..')
import h5py
from rdkit import Chem
from tqdm import tqdm
import argparse

from models.ECloudDecipher.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from models.ECloudDecipher.models.encoding.tokenizers import get_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ecloud_decipher.h5')
    args = parser.parse_args()

    tokenizer = TrieTokenizer(n_seq=64, **get_vocab('mar'))

    with h5py.File(args.data, 'r+') as f:
        ecloud_in = f['eclouds'] 
        smiles_in = f['smiles']   

        smiles = [s.decode('utf-8') for s in smiles_in]
        num_mols = len(smiles)
        print("Number of molecules:", num_mols)

        if 'raw_tokens' in f:
            del f['raw_tokens']
        if 'augmented_tokens' in f:
            del f['augmented_tokens']

        raw_tokens_dset = f.create_dataset('raw_tokens', (num_mols, 64), dtype='int32')
        augmented_tokens_dset = f.create_dataset('augmented_tokens', (num_mols, 64), dtype='int32')

        for i, s in tqdm(enumerate(smiles), total=num_mols):

            raw_token = tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
            aug_token = tokenizer.tokenize_text("[CLIP][UNK][SMILES][SUFFIX][MIDDLE]" + s + "[STOP]", pad=True)

            raw_tokens_dset[i] = raw_token
            augmented_tokens_dset[i] = aug_token
    
    print("Token writing done.")
