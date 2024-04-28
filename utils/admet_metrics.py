import json
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
# from admet_predictor.admet_predictor.inference_self import *
import argparse

sample_source_admet = {'Ames': 0.035663757, 'BBB': 0.50078434, 'Carcinogenicity': 0.2912641, 'CYP1A2-inh': 0.0006484002, 'CYP1A2-sub': 0.106049754, 'CYP2C19-inh': 0.9966728, 'CYP2C19-sub': 0.91121876, 'CYP2C9-inh': 0.99209404, 'CYP2C9-sub': 0.97174656, 'CYP2D6-inh': 0.0063076904, 'CYP2D6-sub': 0.22239795, 'CYP3A4-inh': 0.9991349, 'CYP3A4-sub': 0.9140007, 'DILI': 0.5635481, 'EC': 0.00057776354, 'EI': 0.00041110237, 'F(20%)': 0.7103559, 'F(50%)': 0.2772025, 'FDAMDD': 0.62075585, 'hERG': 0.26188782, 'H-HT': 0.87836444, 'HIA': 0.999379, 'MLM': 4.1524395e-06, 'NR-AhR': 0.0013107283, 'NR-AR': 0.032314595, 'NR-AR-LBD': 0.0005757817, 'NR-Aromatase': 0.012284468, 'NR-ER': 0.033463612, 'NR-ER-LBD': 0.0032456834, 'NR-PPAR-gamma': 0.0013350961, 'Pgp-inh': 0.9998952, 'Pgp-sub': 0.004989651, 'Respiratory': 0.05135913, 'ROA': 0.00913314, 'SkinSen': 0.009424844, 'SR-ARE': 0.390004, 'SR-ATAD5': 4.5936587e-05, 'SR-HSE': 0.00072837755, 'SR-MMP': 0.1396952, 'SR-p53': 0.023079451, 'T12': 0.24679437, 'BCF': 1.4929994, 'Caco-2': -4.629592, 'CL': 7.72671, 'Fu': 1.4939433, 'IGC50': 4.28176, 'LC50': 5.155973, 'LC50DM': 5.607675, 'LogD': 3.446528, 'LogP': 7.30316547262291, 'LogS': -1.179524979501322, 'MDCK': -3.949019014676395, 'PPB': 1.5027597797403995, 'VDss': 3.578444533289475}
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        default=r'G:\tangui\admet_predictor\temp\temp_g.csv'
    )
    parser.add_argument(
        "--modified_features",
        type=List[str],
        default=['LogP', 'LogS', 'MDCK', 'PPB', 'VDss']
    )
    parser.add_argument(
        "--target_admet",
        type=List[float],
        default=sample_source_admet
    )
    parser.add_argument(
        "--k2",
        type=int,
        default=1.0
    )
    args = parser.parse_args()
    return args


# class ControlledGenerationMetrics:
#     """
#     Class to evaluate controlled generation metrics.
#
#
#
#
#     """
#
#     def __init__(self, args) -> None:
#
#         self.classification_features = ['Ames', 'BBB', 'Carcinogenicity', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh',
#                                         'CYP2C19-sub',
#                                         'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-inh',
#                                         'CYP3A4-sub',
#                                         'DILI',
#                                         'EC', 'EI', 'F(20%)', 'F(50%)', 'FDAMDD', 'hERG', 'H-HT', 'HIA', 'MLM',
#                                         'NR-AhR',
#                                         'NR-AR',
#                                         'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'Pgp-inh',
#                                         'Pgp-sub',
#                                         'Respiratory', 'ROA', 'SkinSen', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP',
#                                         'SR-p53',
#                                         'T12']
#         self.regression_features = ['BCF', 'Caco-2', 'CL', 'Fu', 'IGC50', 'LC50', 'LC50DM', 'LogD', 'LogP', 'LogS',
#                                     'MDCK',
#                                     'PPB', 'VDss']
#
#         self.all_features = self.classification_features + self.regression_features
#         self.mol_admets = {}
#         self.valid_smiles = []
#         self.smiles = []
#
#         self.load_regression_stats("./utils/mean_std.json")
#
#         self.load_smiles_and_admets(args.csv_file)  # get self.smiles
#
#         self.get_valid_mols()  # get self.valid_smiles, self.valid_mols
#
#         self.k2 = args.k2  # int, k2 for metrics. If generated moleculars' mean values are in [k2, k1], it is controlled successfully.
#
#         self.modified_features = args.modified_features
#         self.target_admet = self.process_admet(
#             args.target_admet)  # Dict, source admet features(classification features are probs)
#
#         self.metrics = self.evaluate()
#
#     def load_regression_stats(self, file: str):
#         """
#         Load 13 ADMET regression features and stats(mean, std) from a json file.
#         Example:
#             {
#                 "LogP":{
#                     "mean": 0.0,
#                     "std": 1.0,
#                 }
#             }
#         """
#         with open(file, 'r') as f:
#             self.regression_stats = json.load(f)
#
#     def load_smiles_and_admets(self, file: str):
#         """Load ADMET dicts and smiles from csv file using pandas.Smiles is first column, other columns are ADMET features
#
#         Args:
#             file (str): _description_
#         """
#         df = pd.read_csv(file)
#         self.smiles = df.iloc[:, 0].tolist()
#
#         admets = df.iloc[:, 1:].to_dict(orient='records')
#
#         for i, smi in enumerate(self.smiles):
#             self.mol_admets[smi] = self.process_admet(admets[i])
#
#     def is_success(self, feature: str, mol_admet: dict):
#         """Discriminate whether the feature of single mol is controlled successfully.
#
#         Args:
#             feature (str): Controlled feature.
#             mol_admet (dict): Generated single molecular ADMET dict.
#         """
#         if feature in self.classification_features:
#             # feature in classification_features
#             if mol_admet[feature] == self.target_admet[feature]:
#                 return True
#         else:
#             # feature in regression_features
#             if abs(mol_admet[feature] - self.target_admet[feature]) <= self.k2 * self.regression_stats[feature]['std']:
#                 return True
#         return False
#
#     def get_valid_mols(self):
#         """Get valid mols from smiles.
#
#         Args:
#             smiles (List[str]): List of valid SMILES strings.
#         Returns:
#             List[Chem.Mol]: List of mols.
#         """
#         self.valid_mols = []
#         self.valid_smiles = []
#         for smi in self.smiles:
#             try:
#                 mol = Chem.MolFromSmiles(smi)
#                 if self.is_valid(mol):
#                     self.valid_mols.append(mol)
#                     self.valid_smiles.append(smi)
#             except:
#                 continue
#
#     def is_valid(self, mol) -> bool:
#         """Discriminate whether the smiles is valid.
#
#         Args:
#             mol: SMILES string.
#
#         Returns:
#             bool: True if valid, False if invalid.
#         """
#         try:
#             Chem.SanitizeMol(mol)
#         except:
#             return False
#         return True
#
#     def single_mol_success_rate(self, mol_admet: dict):
#         """
#         Calculate success rate of a specific set of features for a single molecular.
#
#         Args:
#             self.target_admet (dict): Target ADMET dict.
#             mol_admet (dict): Generated single molecular ADMET dict.
#             modified_features (List[str]): Specific ADMET features to be modified from target ADMET.
#
#         Return:
#             modified_success_rate (float): Success rate of controlling features.
#             unmodified_success_rate (float): Success rate of uncontrolled features.
#         """
#         num_modified_features = len(self.modified_features)
#         num_unmodified_features = len(self.all_features) - num_modified_features
#
#         count_success_modified = 0
#         count_success_unmodified = 0
#
#         for feature in self.all_features:
#             if feature in self.modified_features:
#                 if self.is_success(feature, mol_admet):
#                     count_success_modified += 1
#             else:
#                 if self.is_success(feature, mol_admet):
#                     count_success_unmodified += 1
#
#         modified_success_rate = count_success_modified / num_modified_features if num_modified_features != 0 else 0
#         unmodified_success_rate = count_success_unmodified / num_unmodified_features if num_unmodified_features != 0 else 0
#         total_success_rate = (count_success_modified + count_success_unmodified) / len(self.all_features)
#
#         return modified_success_rate, unmodified_success_rate, total_success_rate
#
#     def process_admet(self, admet: dict) -> dict:
#         """Convert probs of classification features of ADMET to 0 or 1.
#
#         Args:
#             admet (dict): Dict of 54 unprocessed ADMET features(41 Classification features, 13 Regression features)
#
#         Returns:
#             dict: Dict of 54 processed ADMET features(41 Classification features, 13 Regression features)
#         """
#         for feature in self.classification_features:
#             if admet[feature] >= 0.5:
#                 admet[feature] = 1
#             else:
#                 admet[feature] = 0
#         return admet
#
#     def get_internal_diversity(self, mols) -> float:
#         """Calculate internal diversity of a set of molecules.
#
#         Args:
#             mols (List[Chem.Mol]): List of mols.
#
#         Returns:
#             float: Internal diversity.
#         """
#         similarity = 0
#         fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
#         for i in range(len(fingerprints)):
#             sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
#             similarity += sum(sims)
#         n = len(fingerprints)
#         n_pairs = n * (n - 1) / 2
#         diversity = 1 - similarity / n_pairs
#         return diversity
#
#     def evaluate(self) -> dict:
#         """
#         Evaluate controlling results.
#         Args:
#             self.target_admet(dict):
#                 ADMET target values dict.
#                 e.g. {"LogP": 0.0, "TPSA": 0.0}
#                 len(admet_target.keys()) = 54
#             self.smiles(list):
#                 Generated smiles list.
#                 e.g. ["C", "CC", "CCC"]
#             self.modified_features(list):
#                 Controlled ADMET features.
#                 e.g. ["LogP", "TPSA"]
#             self.k1(List[int]):
#                 k1 for each controlled feature. Control new mean values of generated moleculars we want.
#                 e.g. [1, 1]
#                 new_mean_vals[i] = mean_vals[i] + k1[i] * std_vals[i], (i: idx of contolled feature)
#                 k1[i] should be 0 or 1 for classification features, 0: unchange, 1: change.
#             self.k2(List[int]):
#         Return:
#             dict:
#                 Metrics of controlling results.
#                 "validity"
#                 "internal_diversity"
#                 "modified_features_success_rate":
#                     Average success rate of controlling features for all moleculars.
#                     If 3 features are controlled and for one molecular, 2 features are controlled successfully,
#                     then its ctrl_fea_success_rate is 2/3. Calculate the average of all moleculars.
#                 "unmodified_features_success_rate"
#         """
#         # Initialize metrics.
#         metrics = {
#             # "validity": 0,
#             # "internal_diversity": 0,
#             "modified_features_success_rate": 0,
#             "unmodified_features_success_rate": 0,
#             "all_features_success_rate": 0,
#         }
#         # Initialize counters.
#         num_valid_smiles = len(self.valid_smiles)
#         num_smiles = len(self.smiles)
#         modified_success_rate_list = []
#         unmodified_success_rate_list = []
#         total_success_rate_list = []
#
#         # Calculate metrics.
#         for smi in self.valid_smiles:
#             modified_success_rate, unmodified_success_rate, total_success_rate = self.single_mol_success_rate(
#                 self.mol_admets[smi])
#             modified_success_rate_list.append(modified_success_rate)
#             unmodified_success_rate_list.append(unmodified_success_rate)
#             total_success_rate_list.append(total_success_rate)
#
#         # metrics["validity"] = num_valid_smiles / num_smiles
#         # metrics["internal_diversity"] = self.get_internal_diversity(self.valid_mols)  #
#         metrics["modified_features_success_rate"] = sum(modified_success_rate_list) / num_valid_smiles
#         metrics["unmodified_features_success_rate"] = sum(unmodified_success_rate_list) / num_valid_smiles
#         metrics["all_features_success_rate"] = sum(total_success_rate_list) / num_valid_smiles
#         return metrics

class ControlledGenerationMetrics:
    """
    Class to evaluate controlled generation metrics.




    """

    def __init__(self, args) -> None:

        self.classification_features = ['Ames', 'BBB', 'Carcinogenicity', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh',
                                        'CYP2C19-sub',
                                        'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-inh',
                                        'CYP3A4-sub',
                                        'DILI',
                                        'EC', 'EI', 'F(20%)', 'F(50%)', 'FDAMDD', 'hERG', 'H-HT', 'HIA', 'MLM',
                                        'NR-AhR',
                                        'NR-AR',
                                        'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'Pgp-inh',
                                        'Pgp-sub',
                                        'Respiratory', 'ROA', 'SkinSen', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP',
                                        'SR-p53',
                                        'T12']
        self.regression_features = ['BCF', 'Caco-2', 'CL', 'Fu', 'IGC50', 'LC50', 'LC50DM', 'LogD', 'LogP', 'LogS',
                                    'MDCK',
                                    'PPB', 'VDss']

        self.all_features = self.classification_features + self.regression_features
        self.mol_admets = {}
        self.valid_smiles = []
        self.smiles = []

        self.load_regression_stats("./utils/mean_std.json")

        self.load_smiles_and_admets(args.csv_file)  # get self.smiles

        self.get_valid_mols()  # get self.valid_smiles, self.valid_mols

        self.k2 = args.k2  # int, k2 for metrics. If generated moleculars' mean values are in [k2, k1], it is controlled successfully.

        self.modified_features = args.modified_features
        self.target_admet = self.process_admet(
            args.target_admet)  # Dict, source admet features(classification features are probs)

        self.metrics = self.evaluate()

    def load_regression_stats(self, file: str):
        """
        Load 13 ADMET regression features and stats(mean, std) from a json file.
        Example:
            {
                "LogP":{
                    "mean": 0.0,
                    "std": 1.0,
                }
            }
        """
        with open(file, 'r') as f:
            self.regression_stats = json.load(f)

    def load_smiles_and_admets(self, file: str):
        """Load ADMET dicts and smiles from csv file using pandas.Smiles is first column, other columns are ADMET features

        Args:
            file (str): _description_
        """
        df = pd.read_csv(file)
        self.smiles = df.iloc[:, 0].tolist()

        admets = df.iloc[:, 1:].to_dict(orient='records')

        for i, smi in enumerate(self.smiles):
            self.mol_admets[smi] = self.process_admet(admets[i])

    def is_success(self, feature: str, mol_admet: dict):
        """Discriminate whether the feature of single mol is controlled successfully.

        Args:
            feature (str): Controlled feature.
            mol_admet (dict): Generated single molecular ADMET dict.
        """
        if feature in self.classification_features:
            # feature in classification_features
            if mol_admet[feature] == self.target_admet[feature]:
                return True
        else:
            # feature in regression_features
            if abs(mol_admet[feature] - self.target_admet[feature]) <= self.k2 * self.regression_stats[feature]['std']:
                return True
        return False

    def get_valid_mols(self):
        """Get valid mols from smiles.

        Args:
            smiles (List[str]): List of valid SMILES strings.
        Returns:
            List[Chem.Mol]: List of mols.
        """
        self.valid_mols = []
        self.valid_smiles = []
        for smi in self.smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if self.is_valid(mol):
                    self.valid_mols.append(mol)
                    self.valid_smiles.append(smi)
            except:
                continue

    def is_valid(self, mol) -> bool:
        """Discriminate whether the smiles is valid.

        Args:
            mol: SMILES string.

        Returns:
            bool: True if valid, False if invalid.
        """
        try:
            Chem.SanitizeMol(mol)
        except:
            return False
        return True

    def single_mol_success_rate(self, mol_admet: dict):
        """
        Calculate success rate of a specific set of features for a single molecular.

        Args:
            self.target_admet (dict): Target ADMET dict.
            mol_admet (dict): Generated single molecular ADMET dict.
            modified_features (List[str]): Specific ADMET features to be modified from target ADMET.

        Return:
            modified_success_rate (float): Success rate of modified features.
            unmodified_success_rate (float): Success rate of unmodified features.
            total_success_rate (float): Success rate of all features.
        """
        num_modified_features = len(self.modified_features)
        num_unmodified_features = len(self.all_features) - num_modified_features

        count_success_modified = 0
        count_success_unmodified = 0

        for feature in self.all_features:
            if feature in self.modified_features:
                if self.is_success(feature, mol_admet):
                    count_success_modified += 1
            else:
                if self.is_success(feature, mol_admet):
                    count_success_unmodified += 1

        modified_success_rate = 1 if count_success_modified == num_modified_features else 0
        unmodified_success_rate = 1 if count_success_unmodified == num_unmodified_features else 0
        total_success_rate = modified_success_rate * unmodified_success_rate

        return modified_success_rate, unmodified_success_rate, total_success_rate

    def process_admet(self, admet: dict) -> dict:
        """Convert probs of classification features of ADMET to 0 or 1.

        Args:
            admet (dict): Dict of 54 unprocessed ADMET features(41 Classification features, 13 Regression features)

        Returns:
            dict: Dict of 54 processed ADMET features(41 Classification features, 13 Regression features)
        """
        for feature in self.classification_features:
            if admet[feature] >= 0.5:
                admet[feature] = 1
            else:
                admet[feature] = 0
        return admet

    def get_internal_diversity(self, mols) -> float:
        """Calculate internal diversity of a set of molecules.

        Args:
            mols (List[Chem.Mol]): List of mols.

        Returns:
            float: Internal diversity.
        """
        similarity = 0
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
        for i in range(len(fingerprints)):
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            similarity += sum(sims)
        n = len(fingerprints)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / n_pairs
        return diversity

    def evaluate(self) -> dict:
        """
        Evaluate controlling results.
        Args:
            self.target_admet(dict):
                ADMET target values dict.
                e.g. {"LogP": 0.0, "TPSA": 0.0}
                len(admet_target.keys()) = 54
            self.smiles(list):
                Generated smiles list.
                e.g. ["C", "CC", "CCC"]
            self.modified_features(list):
                Controlled ADMET features.
                e.g. ["LogP", "TPSA"]
            self.k1(List[int]):
                k1 for each controlled feature. Control new mean values of generated moleculars we want.
                e.g. [1, 1]
                new_mean_vals[i] = mean_vals[i] + k1[i] * std_vals[i], (i: idx of contolled feature)
                k1[i] should be 0 or 1 for classification features, 0: unchange, 1: change.
            self.k2(List[int]):
        Return:
            dict:
                Metrics of controlling results.
                "validity"
                "internal_diversity"
                "modified_features_success_rate":
                    Average success rate of controlling features for all moleculars.
                    If 3 features are controlled and for one molecular, 2 features are controlled successfully,
                    then its ctrl_fea_success_rate is 2/3. Calculate the average of all moleculars.
                "unmodified_features_success_rate"
        """
        # Initialize metrics.
        metrics = {
            # "validity": 0,
            # "internal_diversity": 0,
            "modified_features_success_rate": 0,
            "unmodified_features_success_rate": 0,
            "all_features_success_rate": 0,
        }
        # Initialize counters.
        num_valid_smiles = len(self.valid_smiles)
        num_smiles = len(self.smiles)
        modified_success_rate_list = []
        unmodified_success_rate_list = []
        total_success_rate_list = []

        # Calculate metrics.
        for smi in self.valid_smiles:
            modified_success_rate, unmodified_success_rate, total_success_rate = self.single_mol_success_rate(
                self.mol_admets[smi])
            modified_success_rate_list.append(modified_success_rate)
            unmodified_success_rate_list.append(unmodified_success_rate)
            total_success_rate_list.append(total_success_rate)

        # metrics["validity"] = num_valid_smiles / num_smiles
        # metrics["internal_diversity"] = self.get_internal_diversity(self.valid_mols)  #
        metrics["modified_features_success_rate"] = sum(modified_success_rate_list) / num_valid_smiles
        metrics["unmodified_features_success_rate"] = sum(unmodified_success_rate_list) / num_valid_smiles
        metrics["all_features_success_rate"] = sum(total_success_rate_list) / num_valid_smiles
        return metrics

def GEN_ADMET_METRICS(sample_source_admet,
                      csv_file,
                      modified_features,
                      ):
    args = init_args()
    args.csv_file = csv_file
    args.target_admet = sample_source_admet
    args.modified_features = modified_features
    TestMetrics = ControlledGenerationMetrics(args)
    return TestMetrics.metrics


if __name__ == "__main__":
    args = init_args()
    TestMetrics = ControlledGenerationMetrics(args)
    print(TestMetrics.metrics)
