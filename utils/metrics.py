import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import wasserstein_distance


# 有效性mol
def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


# 筛选有效性mol
def get_valid_molecules(molecules):
    valid = []
    for mol in molecules:
        if is_valid(mol):
            valid.append(mol)
    return valid


# mol去重
def get_unique_smiles(valid_molecules):
    unique = set()
    for mol in valid_molecules:
        unique.add(Chem.MolToSmiles(mol))
    return list(unique)


# 对比生成smiles和原始smiles的新颖性
def get_novel_smiles(unique_true_smiles, unique_pred_smiles):
    return list(set(unique_pred_smiles).difference(set(unique_true_smiles)))


# 计算能量
def compute_energy(mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    energy = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0).CalcEnergy()
    return energy


# 对真实mol和生成mol计算wasserstein距离，类似KL散度，不过w距离可以比较离散距离
def wasserstein_distance_between_energies(true_molecules, pred_molecules):
    true_energy_dist = []
    for mol in true_molecules:
        try:
            energy = compute_energy(mol)
            true_energy_dist.append(energy)
        except:
            continue

    pred_energy_dist = []
    for mol in pred_molecules:
        try:
            energy = compute_energy(mol)
            pred_energy_dist.append(energy)
        except:
            continue

    if len(true_energy_dist) > 0 and len(pred_energy_dist) > 0:
        return wasserstein_distance(true_energy_dist, pred_energy_dist)
    else:
        return 0


# 计算多样性
def get_diversity(pred_valid):
    similarity = 0
    pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_valid]
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)
    n = len(pred_fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - similarity / n_pairs
    return diversity


# 计算化学性质
def compute_metrics(pred_molecules, true_molecules):
    if len(pred_molecules) == 0:
        return {
            'validity': 0,
            'uniqueness': 0,
            'novelty': 0,
            'diversity': 0,
            'energies': 0,
        }

    # Passing rdkit.Chem.Sanitize filter
    true_valid = get_valid_molecules(true_molecules)
    pred_valid = get_valid_molecules(pred_molecules)
    validity = len(pred_valid) / len(pred_molecules)

    # Unique molecules
    true_unique = get_unique_smiles(true_valid)
    pred_unique = get_unique_smiles(pred_valid)
    uniqueness = len(pred_unique) / len(pred_valid) if len(pred_valid) > 0 else 0

    # Novel molecules
    pred_novel = get_novel_smiles(true_unique, pred_unique)
    novelty = len(pred_novel) / len(pred_unique) if len(pred_unique) > 0 else 0

    # Difference between Energy distributions
    energies = wasserstein_distance_between_energies(true_valid, pred_valid)

    # diversity
    diversity = get_diversity(pred_valid)
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'energies': energies,
        'diversity': diversity
    }

def get_subgraph_rate(pat_mol_list, pre_mol_list):
    count = 0
    for pre_mol, pat_mol in zip(pre_mol_list, pat_mol_list):
        if len(pre_mol.GetSubstructMatches(pat_mol)) != 0:
            count += 1
    return count/len(pre_mol_list)


# %%
# pred_smiles = ['C', 'CC', 'CCC', 'CCCC', 'N']
# true_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC']
# pred_molecules = []
# true_molecules = []
#
# for smi in pred_smiles:
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         pred_molecules.append(mol)
#     except:
#         pass
# for smi in true_smiles:
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         true_molecules.append(mol)
#     except:
#         pass
#
# compute_metrics(pred_molecules, true_molecules)