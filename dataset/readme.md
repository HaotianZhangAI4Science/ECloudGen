Data Preparation
=======

### Prepare data for **ECloudDiff** from protein ligand dataset. 

Download crossdock_dataset and unzip.

```shell
# modify the data path at generate_pktlig_data.py
# process small data for understanding training process
python 01_dataset/generate_pktlig_data.py --mode train --end 1000 --processor 12
python 01_dataset/generate_pktlig_data.py --mode valid --end 200 --processor 12

# process full data for training 
python 01_dataset/generate_pktlig_data.py --mode train --processor 12
python 01_dataset/generate_pktlig_data.py --mode valid --processor 12
```

For storage estimation, 1000 protein-ligand pairs with h5 format occupy 597 M space. The `generate_pktlig_data.py` firstly saves .npy file separately at the cache path, followed by fusing them into the single h5 file for fast training. You will get `success_indexs.ply`, `cache directory`, and `xxx.h5` file as the output. 

**A common problem of generate_pktlig_data.py is that you do not properly set the xtb command path.** 

### Prepare data for **ECloudDecipher** from smiles txt 

```
# prepare the ecloud from smiles info 
python 02_generate_ligecloud_data.py --data/demo.smi --output data/ecloud_decipher.h5 --num_workers 8

# after obtaining the ecloud, tokenize the corresponding smiles
python 02_tokenize.py --data data/ecloud_decipher.h5
```

I provided a demo dataset (800 valid SMILES and 1 invalid SMILES) to help you better understand the data preparation process. For a larger dataset, such as CHEBLE, please provide the `all.smi` file containing the molecules. Typically, I split the `all.smi` file into chunks of 100k SMILES and then run the H5 file creation job. After obtaining the chunked H5 files, you can merge them with the following code:

```python
import h5py
import numpy as np

def merge_h5_files(file_list, output_file):
    with h5py.File(output_file, 'w') as fout:
        all_smiles = []
        all_eclouds = []

        for file in file_list:
            with h5py.File(file, 'r') as fin:
                all_smiles.extend(fin['smiles'][:])
                all_eclouds.extend(fin['eclouds'][:])

        fout.create_dataset("smiles", data=np.array(all_smiles, dtype='S'))
        fout.create_dataset("eclouds", data=np.array(all_eclouds))

    print(f"Merged {len(file_list)} files into {output_file}")

h5_files = ["batch1.h5", "batch2.h5", "batch3.h5"]
merge_h5_files(h5_files, "final_merged.h5")
```

