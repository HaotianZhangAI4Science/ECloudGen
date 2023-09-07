ECloudGen
=======


## Environment 

### Install via conda yaml file (cuda 11.3)

```python
mamba env create -f resgen.yml
mamba activate resgen 
```

### Install manually 

This environment have been successfully tested on CUDA==11.3

```
mamba create -n ecloud rdkit python=3.9 h5py jupyter ipykernel -c conda-forge
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.26.1 adapter-transformers==3.2.1 pytorch-ignite
```

<div style="text-align: center;">
    <figure style="width: 30%; display: inline-block;">
        <img src='./figs/ligink.gif' alt="Image 1" style="width: 100%;">
        <figcaption>Latent Diffusion Process</figcaption>
    </figure>
    <figure style="width: 30%; display: inline-block;">
        <img src='./figs/liglava.gif' alt="Image 2" style="width: 100%;">
        <figcaption>Electron Clouds</figcaption>
    </figure>
    <figure style="width: 30%; display: inline-block;">
        <img src='./figs/pkt_lig.gif' alt="Image 3" style="width: 100%;">
        <figcaption>Pocket-Liand Interaction</figcaption>
    </figure>
</div>

















## Data 

The main data for training is CrossDock2020, which is utilized by most of the methods. 

**Note: data is only necessary for training. For use-only mode, please directly check the generation part.**  

#### (Optional) Download the data from the original source

```python
wget https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.1.tgz -P data/crossdock2020/
tar -C data/crossdock2020/ -xzf data/crossdock2020/CrossDocked2020_v1.1.tgz
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_train0_fixed.types -P data/crossdock2020/
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_test0_fixed.types -P data/crossdock2020/
```

The storage size of original crossdock2020 is 50 GB, hard to download and unzip. You can skip to the Approach 1 or Approach 2 for training preparation. 

#### Approach 1: Download the Pocket Data for Processing

You can download the processed data from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM). This is the processed version of original files, which is processed by [Luoshi Tong](https://github.com/luost26/3D-Generative-SBDD/tree/main/data).

Note: [index.pkl](https://github.com/HaotianZhangAI4Science/ResGen/tree/main/data/crossdocked_pocket10),  [split_by_name.pt](https://github.com/HaotianZhangAI4Science/ResGen/tree/main/data). are automatically downloaded with the SurfGen code.  index.pkl saves the information of each protein-ligand pair, while split_by_name.pt save the train-test split of the dataset.

```python
tar -xzvf crossdocked_pocket10.tar.gz
python process_data.py --raw_data ./data/crossdocked_pocket10 
```

#### Approach 2: Download the Processed Data

or you can download the processed data [lmdb](https://doi.org/10.5281/zenodo.7759114), [key](https://doi.org/10.5281/zenodo.7759114), and [name2id](https://doi.org/10.5281/zenodo.7759114). 



# Generation

The trained model's parameters could be downloaded [here](https://drive.google.com/file/d/1bUBNDNc0ZzcG4WgY18aQB0PEVOO6RRQQ/view?usp=share_link).  

```python
python gen.py --pdb_file ./examples/4iiy.pdb --sdf_file ./examples/4iiy_ligand.sdf --outdir ./examples
```

You can also follow the guide at generation/generation.ipynb 

We provide the pdbid-14gs as the example

<div align=center>
<img src="./figures/example.png" width="50%" height="50%" alt="TOC" align=center />
</div>



# Training 

The training process is released as train.py, the following command is an example of how to train a model.

```python
python train.py --config ./configs/train_res.yml --logdir logs
```



# Acknowledge

This project draws in part from [GraphBP](https://github.com/divelab/GraphBP) and [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol), supported by GPL-v3 License and MIT License. Thanks for their great work and code, hope readers of interest could check their work, too.  









