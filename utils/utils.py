import argparse
import os
import warnings
import torch
import shutil
import copy
# from einops import repeat
import numpy as np
from rdkit.Chem import rdMolTransforms, AllChem
from rdkit import Chem
from math import ceil, pi
import random
import copy
import math
from scipy.spatial.distance import cdist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(os.getcwd(), fn))

def _seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
    _seed_everything(seed)


def move_tokenizer(cfg, save_path):
    if not os.path.exists(os.path.join(save_path, 'tokenizer.json')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'tokenizer.json'),
                    os.path.join(save_path, 'tokenizer.json'))
    if not os.path.exists(os.path.join(save_path, 'merges.txt')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'merges.txt'),
                    os.path.join(save_path, 'merges.txt'))
    if not os.path.exists(os.path.join(save_path, 'vocab.json')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'vocab.json'),
                    os.path.join(save_path, 'vocab.json'))

def save_config(cfg, model):
    save_path = os.path.join('save', cfg.save, 'model_config.json')
    if os.path.exists(save_path):
        return
    model.config.to_json_file(save_path)

def get_parameter_number(model):
    """
    Calculate the total number of parameters and the number of trainable parameters in a given model,
    and convert these numbers to millions (M).

    Args:
        model (torch.nn.Module): The neural network model to evaluate.

    Returns:
        dict: A dictionary with keys 'Total' and 'Trainable', where the numbers are expressed in millions.
    """
    # Calculate the total number of parameters and convert to millions
    total_num = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Calculate the number of trainable parameters and convert to millions
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    return {'### Molde Paramaters: Total': f'{total_num:.2f}M', 'Trainable': f'{trainable_num:.2f}M ###'}

def accuracy(outputs, targets, ignore=-100, use_label=False):
    if use_label:
        # TODO: hard coding
        targets[:, :6] = -100

    _, pred = outputs.topk(5, -1, True, True)
    targets_len = (targets != ignore).sum(-1)
    ignore_len = (targets == ignore).sum(-1)

    targets = repeat(targets, 'b l -> b l p', p=5)
    pred[targets == -100] = -100

    res = []
    for k in [1, 5]:
        correct = (pred.eq(targets)[..., :k].sum(-1) >= 1).sum(-1)

        acc = ((correct - ignore_len) / targets_len).mean()
        res.append(acc)

    return res[0], res[1]

def accuracy2(outputs, targets, ignore=-100):
    mask = targets.ne(ignore)
    pred_id = outputs[mask].argmax(-1)
    targets = targets[mask]
    masked_hit = (pred_id == targets).long().sum()
    masked_cnt = mask.long().sum()
    hit_rate = masked_hit/masked_cnt
    return hit_rate


def get_mol_centroid(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = np.mean(conformer.GetPositions(), axis=0)
    return centroid


def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation


def centralize(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = get_mol_centroid(mol, confId)
    translation = trans(-centroid[0], -centroid[1], -centroid[2])
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol



def generate_sdf_atomic_labels(mol, Atomic2id, resolution=0.5, max_dist=6.75):
    # 获取分子坐标和原子类型
    coords = mol.GetConformer().GetPositions()
    atoms = np.array([str(atom.GetSymbol()) for atom in mol.GetAtoms()])

    # 判断网格是否合法
    size = math.ceil(2 * max_dist / resolution + 1)
    assert size % 1 == 0
    size = int(size)

    # 计算分子坐标在网格的位置 -> 网格坐标
    grid_coords = ((coords + max_dist) / resolution).round().astype(int)
    in_box = ((grid_coords >= 0) & (grid_coords < size)).all(axis=1)
    grid_coords = grid_coords[in_box]

    # 计算每个格点到最近坐标点的距离
    distances_matrix = cdist(grid_coords, np.indices((size, size, size)).reshape(3, -1).T, 'euclidean')
    distances = np.min(distances_matrix, axis=0)
    atomic = atoms[np.argmin(distances_matrix, axis=0)]
    vfunc = np.vectorize(Atomic2id.get)
    atom_id = vfunc(atomic)

    # 将距离填充到张量中
    # distan_tensor = distances.reshape(size, size, size) / (np.sqrt(3) * size)
    distan_tensor = distances.reshape(size, size, size)
    atomic_tensor = atom_id.reshape(size, size, size)

    return distan_tensor, atomic_tensor, grid_coords

def generate_sdf_labels(coords, resolution=0.5, max_dist=6.75):
    size = 2 * max_dist / resolution + 1
    assert size % 1 == 0
    size = int(size)
    # 将坐标范围映射到0到27之间
    coords = ((coords + max_dist) / resolution).round().astype(int)
    coords = np.clip(coords, 0, size - 1)

    # 生成一个28x28x28的全0张量
    tensor = np.zeros((size, size, size))

    # 计算每个格点到最近坐标点的距离
    distances = cdist(coords, np.indices((size, size, size)).reshape(3, -1).T, 'euclidean')
    distances = np.min(distances, axis=0)

    # 将距离填充到张量中
    tensor = distances.reshape(size, size, size)
    tensor = tensor / (np.sqrt(3) * max_dist * 2)

    return tensor


def set_mol_position(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def get_optimizer(cfg, model):
    # Define parameter groups with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_BIAS, 'lr': cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR}
    ]

    # Create the optimizer
    if cfg.SOLVER.OPTIMIZER_NAME.lower() == "adamw":
        print("Use AdamW optimizer with learning rate:", cfg.SOLVER.BASE_LR)
        return AdamW(optimizer_grouped_parameters,
                     lr=cfg.SOLVER.BASE_LR,
                     betas=(0.9, 0.99),
                     eps=1e-6)
    else:
        raise NotImplementedError(f"Optimizer not supported: {cfg.SOLVER.OPTIMIZER_NAME}")


def get_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHED.lower() == "warmuplinearlr":
        num_training_steps = cfg.SOLVER.MAX_STEPS
        num_warmup_steps = int(cfg.SOLVER.WARMUP_STEP_RATIO * num_training_steps)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                cfg.SOLVER.WARMUP_FACTOR,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        print("Use WarmupLinearLR scheduler with warmup steps:", num_warmup_steps)
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError(f"Scheduler not supported: {cfg.SOLVER.SCHED}")
    
def get_scheduler_dataloader(cfg, optimizer, dataloader):
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.SOLVER.GRADIENT_ACC)
    max_train_steps = cfg.SOLVER.MAX_EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.SOLVER.WARMUP_STEP_RATIO * max_train_steps)

    if cfg.SOLVER.SCHED == "WarmupLinearLR":
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(max_train_steps - current_step) / float(max(1, max_train_steps - num_warmup_steps))
            )
        print("Use WarmupLinearLR scheduler with warmup steps:", num_warmup_steps)
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError(f"Scheduler not supported: {cfg.SOLVER.SCHED}")