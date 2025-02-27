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
    parser.add_argument('--output', type=str, default='data/ecloud_decipher_token.h5')
    args = parser.parse_args()

    tokenizer = TrieTokenizer(n_seq=64, **get_vocab('mar'))

    with h5py.File(args.data, 'r') as f_in:  
        ecloud_in = f_in['eclouds']
        smiles_in = f_in['smiles']   
        smiles = [s.decode('utf-8') for s in smiles_in]

        print("Number of molecules:", len(smiles))

        with h5py.File(args.output, 'w') as f_out:
            eclouds_out = f_out.create_dataset("eclouds", (len(smiles), 32, 32, 32), dtype='float32')
            raw_tokens_out = f_out.create_dataset("raw_tokens", (len(smiles), 64), dtype='int32')
            augmented_tokens_out = f_out.create_dataset("augmented_tokens", (len(smiles), 64), dtype='int32')

            for i, s in tqdm(enumerate(smiles), total=len(smiles)):
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                s_norm = Chem.MolToSmiles(mol)

                raw_token = tokenizer.tokenize_text("[SMILES]" + s_norm + "[STOP]", pad=True)
                aug_token = tokenizer.tokenize_text("[CLIP][UNK][SMILES][SUFFIX][MIDDLE]" + s_norm + "[STOP]", pad=True)

                eclouds_out[i] = ecloud_in[i]
                raw_tokens_out[i] = raw_token
                augmented_tokens_out[i] = aug_token

                if i % 10000 == 0:
                    print("Processed:", i)