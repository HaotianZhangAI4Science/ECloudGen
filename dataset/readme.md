Data Preparation
=======

Prepare data for ECloudDiff from protein ligand dataset. 

Download crossdock_dataset and unzip.

```shell
# modify the data path at generate_pktlig_data.py
# process small data for understanding training process
python dataset/generate_pktlig_data.py --mode train --end 1000 --processor 12
python dataset/generate_pktlig_data.py --mode valid --end 200 --processor 12

# process full data for training 
python dataset/generate_pktlig_data.py --mode train --processor 12
python dataset/generate_pktlig_data.py --mode valid --processor 12
```

For storage estimation, 1000 protein-ligand pairs with h5 format occupy 597 M space. The `generate_pktlig_data.py` firstly saves .npy file separately at the cache path
