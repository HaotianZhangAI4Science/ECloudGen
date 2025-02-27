""""
Module defining the main Particle Swarm optimizer class.
"""
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
from rdkit import Chem, rdBase
from .swarm import Swarm
import torch
import warnings
warnings.filterwarnings('ignore', message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

def canonicalize_smiles(sml):
    """
    Function that canonicalize a given SMILES
    :param sml: input SMILES
    :return: The canonical version of the input SMILES
    """
    mol = Chem.MolFromSmiles(sml)
    if mol is not None:
        sml = Chem.MolToSmiles(mol)
    return sml

class BasePSOptimizer:
    """
        Base particle swarm optimizer class. It handles the optimization of a swarm object.
    """
    def __init__(self, swarms, inference_model, scoring_functions=None, device='cpu', voice=True):
        """

        :param swarms: List of swarm objects each defining an individual particle swarm that
            is used for optimization.
        :param inference_model: The inference model used to encode/decode smiles to/from the
            Continuous Data-Diven molecular Descriptor (CDDD) space.
        :param scoring_functions: List of functions that are used to evaluate a generated molecule.
            Either take a RDKit mol object as input or a point in the cddd space.
        """
        self.infer_model = inference_model
        self.scoring_functions = scoring_functions
        self.device = device
        self.swarms = swarms
        self.voice = voice
        self.best_solutions = pd.DataFrame(columns=["smiles", "fitness"])
        self.best_fitness_history = pd.DataFrame(columns=["step", "swarm", "fitness"])

    def update_fitness(self, swarm):
        """
        Method that calculates and updates the fitness of each particle in  a given swarm. A
        particles fitness is defined as weighted average of each scoring functions output for
        this particle.
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        assert self.scoring_functions is not None
        total_fitness = torch.zeros(len(swarm.smiles), device=self.device)
        mol_list = [Chem.MolFromSmiles(sml) for sml in swarm.smiles]
        
        for scoring_function in self.scoring_functions:
            if scoring_function.is_mol_func:
                unscaled_scores, scaled_scores, desirability_scores = scoring_function(mol_list)
            else:
                unscaled_scores, scaled_scores, desirability_scores = scoring_function(swarm.x)
            swarm.unscaled_scores[scoring_function.name] = torch.tensor(unscaled_scores, device=self.device)
            swarm.scaled_scores[scoring_function.name] = torch.tensor(scaled_scores, device=self.device)
            swarm.desirability_scores[scoring_function.name] = torch.tensor(desirability_scores, device=self.device)

            total_fitness += torch.tensor(scaled_scores, device=self.device) 

        total_fitness /= sum(sf.weight for sf in self.scoring_functions)

        swarm.update_fitness(total_fitness)
        return swarm

    def _next_step_and_evaluate(self, swarm):
        """
        Method that wraps the update of the particles position (next step) and the evaluation of
        the fitness at these new positions.
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        swarm.next_step() # Update particle positions
        smiles = self.infer_model.emb_to_seq(swarm.x)  # Decode embeddings to SMILES
        swarm.smiles = smiles
        swarm.x = self.infer_model.seq_to_emb(swarm.smiles)
        swarm = self.update_fitness(swarm)
        return swarm

    def _update_best_solutions(self, num_track):
        """
        Method that updates the best_solutions dataframe that keeps track of the overall best
        solutions over the course of the optimization.
        :param num_track: Length of the best_solutions dataframe.
        :return: The max, min and mean fitness of the best_solutions dataframe.
        """
    # Collect all current swarm solutions into a single DataFrame
        new_data = {
            'smiles': [sml for swarm in self.swarms for sml in swarm.smiles],
            'fitness': [fit.item() for swarm in self.swarms for fit in swarm.fitness]
        }
        new_df = pd.DataFrame(new_data)
        new_df.smiles = new_df.smiles.map(canonicalize_smiles)
        self.best_solutions = pd.concat([self.best_solutions, new_df], ignore_index=True)
        self.best_solutions.drop_duplicates(subset="smiles", inplace=True)
        self.best_solutions.sort_values(by="fitness", ascending=False, inplace=True)
        # Retain only the top num_track entries
        self.best_solutions = self.best_solutions.head(num_track)

        # Extract statistics
        best_solutions_max = self.best_solutions['fitness'].max()
        best_solutions_min = self.best_solutions['fitness'].min()
        best_solutions_mean = self.best_solutions['fitness'].mean()
        return best_solutions_max, best_solutions_min, best_solutions_mean

    def _update_best_fitness_history(self, step):
        """
        tracks best solutions for each swarm
        :param step: The current iteration step of the optimizer.
        :return: None
        """
        data = {
            'step': [step] * len(self.swarms),
            'swarm': list(range(len(self.swarms))),
            'fitness': [swarm.swarm_best_fitness.item() for swarm in self.swarms],
            'smiles': [swarm.best_smiles for swarm in self.swarms]
        }
        new_df = pd.DataFrame(data)
        self.best_fitness_history = pd.concat([self.best_fitness_history, new_df], ignore_index=True)


    def run(self, num_steps, num_track=10):
        """
        The main optimization loop.
        :param num_steps: The number of update steps.
        :param num_track: Number of best solutions to track.
        :return: The optimized particle swarm.
        """
        # evaluate initial score
        print('Particle Swarm Optimization...')
        for swarm in self.swarms:
            self.update_fitness(swarm)
        for step in range(num_steps):
            self._update_best_fitness_history(step)
            max_fitness, min_fitness, mean_fitness = self._update_best_solutions(num_track)
            if self.voice:
                print("Step %d, max: %.3f, min: %.3f, mean: %.3f"
                    % (step, max_fitness, min_fitness, mean_fitness))
            for swarm in self.swarms:
                self._next_step_and_evaluate(swarm)
        if self.voice:
            print('Particle Swarm Optimization finished.')
            print('You can access the best solutions with the self.best_solutions and self.best_fitness_history.')
        
        return self.swarms

    @classmethod
    def from_query(cls, init_smiles, num_part, num_swarms, inference_model,
                   scoring_functions=None, phi1=2., phi2=2., phi3=2., x_min=-1.,
                   x_max=1., v_min=-0.6, v_max=0.6, device='cpu', **kwargs):
        """
        Classmethod to create a PSO instance with multiple swarms which particles are initialized at
        the position of the embedded input SMILES.
        """
        embedding = inference_model.seq_to_emb(init_smiles)
        swarms = [
            Swarm.from_query(
                init_sml=init_smiles,
                init_emb=embedding,
                num_part=num_part,
                v_min=v_min,
                v_max=v_max,
                x_min=x_min,
                x_max=x_max,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3,
                device=device) for _ in range(num_swarms)]
        return cls(swarms, inference_model, scoring_functions, device=device,**kwargs)

    @classmethod
    def from_query_list(cls, init_smiles, num_part, num_swarms, inference_model,
                        scoring_functions=None, phi1=2., phi2=2., phi3=2., x_min=-1.,
                        x_max=1., v_min=-0.6, v_max=0.6, device='cpu',**kwargs):
        """
        Classmethod to create a PSO instance with multiple swarms which particles are initialized at
        the position of the embedded input SMILES.
        """
        assert isinstance(init_smiles, list)
        assert len(init_smiles) == num_swarms
        embedding = inference_model.seq_to_emb(init_smiles)
        swarms = []
        for i, sml in enumerate(init_smiles):
            swarms.append(Swarm.from_query(
                init_sml=sml,
                init_emb=embedding[i],
                num_part=num_part,
                v_min=v_min,
                v_max=v_max,
                x_min=x_min,
                x_max=x_max,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3,
                device=device))

        return cls(swarms, inference_model, scoring_functions, device=device,**kwargs)

    @classmethod
    def from_swarm_dicts(cls, swarm_dicts, inference_model, scoring_functions=None, x_min=-1., x_max=1.,
                         inertia_weight=0.9, phi1=2., phi2=2., phi3=2., device='cpu', **kwargs):
        """
        Classmethod to create a PSO instance from a list of swarm dictionaries. This can be used to
        reinitialize a PSO instance from a saved state.
        """
        swarms = [Swarm.from_dict(
            dictionary=swarm_dict,
            x_min=x_min,
            x_max=x_max,
            inertia_weight=inertia_weight,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            device=device
        ) for swarm_dict in swarm_dicts]
        return cls(swarms, inference_model, scoring_functions, device=device,**kwargs)

    def __getstate__(self):
        """dont pickle all swarms --> faster serialization/multiprocessing"""
        return {k: v for k, v in self.__dict__.items() if k not in ('swarms',)}


class ParallelSwarmOptimizer(BasePSOptimizer):
    def _next_step_and_evaluate(self):
        """
        c
        the fitness at these new positions.
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        num_part = self.swarms[0].num_part
        emb = []
        for swarm in self.swarms:
            swarm.next_step()
            emb.append(swarm.x)
        emb = np.concatenate(emb)
        smiles = self.infer_model.emb_to_seq(emb)
        x = self.infer_model.seq_to_emb(smiles)
        for i, swarm in enumerate(self.swarms):
            swarm.smiles = smiles[i*num_part: (i+1)*num_part]
            swarm.x = x[i*num_part: (i+1)*num_part]
            swarm = self.update_fitness(swarm)

    def run(self, num_steps, num_track=10):
        """
        The main optimization loop.
        :param num_steps: The number of update steps.
        :param num_track: Number of best solutions to track.
        :return: The optimized particle swarm.
        """
        # evaluate initial score
        for swarm in self.swarms:
            self.update_fitness(swarm)
        for step in range(num_steps):
            self._update_best_fitness_history(step)
            max_fitness, min_fitness, mean_fitness = self._update_best_solutions(num_track)
            if self.voice:
                print("Step %d, max: %.3f, min: %.3f, mean: %.3f"
                    % (step, max_fitness, min_fitness, mean_fitness))
            self._next_step_and_evaluate()
        return self.swarms, self.best_solutions



class MPPSOOptimizer(BasePSOptimizer):
    """
    A PSOOptimizer class that uses multiprocessing to parallelize the optimization of multiple
    swarms. Only works if the inference_model is a instance of the inference_server class in the
    CDDD package that rolls out calculations on multiple zmq servers (possibly on multiple GPUs).
    """
    # TODO: this is different from the base class, as run() does no initial evaluation but got the evaluate query method.
    def __init__(self, swarms, inference_model, scoring_functions=None, num_workers=1):
        """
        :param swarms: List of swarm objects each defining an individual particle swarm that is
            used for optimization.
        :param inference_model: The inference model used to encode/decode smiles to/from the
            Continuous Data-Diven molecular Descriptor (CDDD) space. Should be an inference_server
            instance to benefit from multiprocessing.
        :param scoring_functions: List of functions that are used to evaluate a generated molecule.
            Either take a RDKit mol object as input or a point in the cddd space.
        :param num_workers: Number of workers used for the multiprocessing.
        """
        super().__init__(swarms, inference_model, scoring_functions)
        self.num_workers = num_workers

    def evaluate_query(self):
        pool = mp.Pool(self.num_workers)
        self.swarms = pool.map(self.update_fitness, self.swarms)
        pool.close()
        return self.swarms

    def run(self, num_steps, num_track=500):
        """
        The main optimization loop in the multiprocessing case with a bit more result
        tracking and timing.
        :param num_steps: The number of update steps.
        :param num_track: Number of best solutions to track.
        :return:
            swarms: The optimized particle swarm.
            best_solutions: The best solutions found over the course of optimization.
        """
        pool = mp.Pool(self.num_workers)
        for step in range(num_steps):
            start_time = time.time()
            self.swarms = pool.map(self._next_step_and_evaluate, self.swarms)
            end_time = time.time() - start_time
            max_fitness, min_fitness, mean_fitness = self._update_best_solutions(num_track)
            self._update_best_fitness_history(step)
            print("Step %d, max: %.3f, min: %.3f, mean: %.3f, et: %.1f s"
                  %(step, max_fitness, min_fitness, mean_fitness, end_time))
            if (num_track == 1) & (self.best_solutions[:num_track].fitness.mean() == 1.):
                break
            elif self.best_solutions[:num_track].fitness.mean() == 1.:
                break
        pool.close()
        return self.swarms, self.best_solutions

class MPPSOOptimizerManualScoring(MPPSOOptimizer):
    def __init__(self, swarms, inference_model, num_workers=1):
        super().__init__(swarms, inference_model, num_workers=num_workers)

    def _next_step_and_evaluate(self, swarm, fitness):
        """
        Method that updates the particles position (next step)
        :param swarm: The swarm that is updated.
        :return: The swarm that is updated.
        """
        swarm.update_fitness(fitness)
        swarm.next_step()
        smiles = self.infer_model.emb_to_seq(swarm.x)
        swarm.smiles = smiles
        swarm.x = self.infer_model.seq_to_emb(swarm.smiles)
        return swarm

    def run_one_iteration(self, fitness):
        pool = mp.Pool(self.num_workers)
        self.swarms = pool.starmap(self._next_step_and_evaluate, zip(self.swarms, fitness))
        pool.close()
        return self.swarms
