import h5py
from rdkit import Chem
from models.ECloudDecipher.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from models.ECloudDecipher.models.encoding.tokenizers import get_vocab
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, default='data/ecloud.h5')
    args.add_argument('--smiles', type=str, default='data/all.smi')
    args.add_argument('--output', type=str, default='data/ecloud_coati.h5')
    args = v.parse_args()

    f = h5py.File(args.data, 'r')
    ecloud_item = f['eclouds']
    with open(args.smiles) as f:
            smiles=[line.strip('\n') for line in f]
    print('demo_data len: ', len(smiles))

    tokenizer = TrieTokenizer(n_seq=64, **get_vocab('mar'))
    with h5py.File(args.output,'w') as f_:
            eclouds=f_.create_dataset("eclouds", (len(smiles),32,32,32), dtype='f')
            raw_tokens=f_.create_dataset("raw_tokens", (len(smiles),64), dtype='i')
            augmented_tokens=f_.create_dataset("augmented_tokens", (len(smiles),64), dtype='i')
            for i, s in tqdm(enumerate(smiles)):
                    s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    raw_token = tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                    augmented_token = tokenizer.tokenize_text("[CLIP][UNK][SMILES][SUFFIX][MIDDLE]" + s + "[STOP]", pad=True)
                    eclouds[i] = ecloud_item[i]
                    raw_tokens[i] = raw_token
                    augmented_tokens[i] = augmented_token
                    if i % 100000 == 0:
                            print('process', i)

    f.close()


