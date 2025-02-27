import random
import torch

class Swarm:
    """
    Class that defines a Swarm that can be optimized by a PSOptimizer. Most PSO calculations are
    done in here.
    """
    def __init__(self, smiles, x, v, x_min=-1., x_max=1.,
                 inertia_weight=0.9, phi1=2., phi2=2., phi3=2.,device='cpu'):
        """
        :param smiles: The SMILES that define the molecules at the positions of the particles of
            the swarm.
        :param x: The position of each particle in the swarm (smiles = decoder(x))
        :param v: The velocity of each particle in the swarm.
        :param x_min: min bound of the optimization space (should be set to -1 as its the default
            CDDD embeddings take values between -1 and 1.
        :param x_max: max bound of the optimization space (should be set to -1 as its the default
            CDDD embeddings take values between -1 and 1.
        :param inertia_weight: PSO hyperparamter.
        :param phi1: PSO hyperparamter.
        :param phi2: PSO hyperparamter.
        :param phi3: PSO hyperparamter.
        """
        self.smiles = smiles
        self.device = device
        self.x = x.clone().detach()
        self.v = v.clone().detach()
        # self.x = torch.tensor(x, dtype=torch.float32, device=device)
        # self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x_min = x_min
        self.x_max = x_max
        self.inertia_weight = inertia_weight
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.num_part = len(smiles)

        self.unscaled_scores = {}
        self.scaled_scores = {}
        self.desirability_scores = {}
        self.fitness = torch.zeros(self.num_part, device=device)
        self.swarm_best_x = self.x.clone()
        self.particle_best_x = self.x.clone()
        self.history_swarm_best_x = [self.x.clone()]
        self.swarm_best_fitness = torch.tensor(0.0, device=device)
        self.particle_best_fitness = self.fitness.clone()
        self.best_smiles = self.smiles[0]

    def next_step(self):
        """
        Method to update the positions of the particle in the swarm.
        :return: None
        """
        u1 = torch.rand(self.num_part, 1, device=self.device) * self.phi1
        u2 = torch.rand(self.num_part, 1, device=self.device) * self.phi2
        u3 = torch.rand(self.num_part, 1, device=self.device) * self.phi3
        random_hist_idx = random.choice(range(len(self.history_swarm_best_x)))
        v_u1 = u1 * (self.particle_best_x - self.x)
        v_u2 = u2 * (self.swarm_best_x - self.x)
        v_u3 = u3 * (self.history_swarm_best_x[random_hist_idx] - self.x)
        self.v = self.inertia_weight * self.v + v_u1 + v_u2 + v_u3
        self.x += self.v
        self.x = torch.clamp(self.x, self.x_min, self.x_max) 

    def update_fitness(self, fitness):
        """
        Updates the fitness of the particles in the swarm. Also updates swarm properties such as
        the best positions (highest fitness) of the whole swarm and each individual particle.
        :param fitness: the fitness of each particle at the new position.
        :return: None
        """
        self.fitness = fitness.clone().detach()
        best_idx = torch.argmax(self.fitness)
        best_fitness = fitness[best_idx]
        if best_fitness > self.swarm_best_fitness:
            self.history_swarm_best_x.append(self.x[best_idx].clone())
            self.swarm_best_fitness = best_fitness.clone()
            self.swarm_best_x = self.x[best_idx].clone()
            self.best_smiles = self.smiles[best_idx]
        updated_best_mask = fitness > self.particle_best_fitness
        self.particle_best_x = torch.where(updated_best_mask.unsqueeze(1), self.x, self.particle_best_x)
        self.particle_best_fitness = torch.where(updated_best_mask, self.fitness, self.particle_best_fitness)

    def __repr__(self):
        return 'mso.swarm.Swarm num_part={} best_fitness={}'.format(self.num_part,
                                                                    self.swarm_best_fitness)

    @classmethod
    def from_dict(cls, dictionary, x_min=-1., x_max=1.,
                 inertia_weight=0.9, phi1=2., phi2=2., phi3=2., device='cpu'):
        """
        Classmethod to create a Swarm instance from a dictionary. Can be used to reinitialize a
        Swarm with all important properties.
        :param dictionary: Dictionary with swarm parameters.
        :param x_min: min bound of the optimization space (should be set to -1 as its the default
            CDDD embeddings take values between -1 and 1.
        :param x_max: max bound of the optimization space (should be set to -1 as its the default
            CDDD embeddings take values between -1 and 1.
        :param inertia_weight: PSO hyperparamter.
        :param phi1: PSO hyperparamter.
        :param phi2: PSO hyperparamter.
        :param phi3: PSO hyperparamter.
        :return: A Swarm instance.
        """
        particles = dictionary['particles']
        smiles = [particle['smiles'] for particle in particles]
        dscore = torch.tensor([particle['dscore'] for particle in particles], dtype=torch.float32, device=device)
        position = torch.tensor([particle['x'] for particle in particles], dtype=torch.float32, device=device)
        velocity = torch.tensor([particle['v'] for particle in particles], dtype=torch.float32, device=device)
        particle_best_x = torch.tensor([particle['part_best_x'] for particle in particles], dtype=torch.float32, device=device)
        particle_best_fitness = torch.tensor([particle['part_best_fitness'] for particle in particles], dtype=torch.float32, device=device)

        swarm = Swarm(
            smiles=smiles,
            x=position,
            v=velocity,
            x_min=x_min,
            x_max=x_max,
            inertia_weight=inertia_weight,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            device=device
        )

        swarm.particle_best_x = particle_best_x
        #swarm.unscaled_scores = {score['name']:  score['unscaled'] for score in scores}
        #swarm.scaled_scores = {score['name']: score['scaled'] for score in scores}
        swarm.fitness = dscore
        swarm.history_swarm_best_x = [torch.tensor(el, dtype=torch.float32, device=device) for el in dictionary['best_positions']]
        swarm.swarm_best_fitness = dictionary['best_fitness']
        swarm.particle_best_fitness = particle_best_fitness
        return swarm

    @classmethod
    def from_query(cls, init_sml, init_emb, num_part, v_min=-0.6, v_max=0.6, device='cpu', *args, **kwargs):
        """
        Classmethod to create a new Swarm instance from a single query. All particles are
        initialized at the same defined position.
        :param init_sml: The initial SMILES that defines the starting point of the particles in
            the swarm. If it is a list of multiple smiles, num_part smiles will be randomly drawn
        :param init_emb: The initial position of the particles in the swarm
            (init_emb = encoder(init_smiles)
        :param num_part: The number of particles that are initialized in the swarm at the given
            initial position.
        :param v_min: The lower bound of the uniform distribution used to sample the initial
            velocity.
        :param v_max: The upper bound of the uniform distribution used to sample the initial
            velocity.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A Swarm instance.
        """
        if isinstance(init_sml, list):
            idxs = torch.randint(0, len(init_sml), (num_part,), device=device)
            smiles = [init_sml[i] for i in idxs]
            x = init_emb[idxs].to(device) 
        else:
            smiles = num_part * [init_sml]
            x = torch.tile(init_emb, (num_part, 1))
        v = (torch.rand((num_part, init_emb.shape[-1]), device=device) * (v_max - v_min) + v_min)
        swarm = Swarm(smiles=smiles, x=x, v=v, device=device,*args, **kwargs)
        return swarm

    def to_dict(self):
        """
        Method to write the swarm with all its properties to a dictionary. This dictionary can be
        used to reinitialize the swarm.
        :return: A dictionary with all swarm properties.
        """
        particles = []
        for i in range(self.num_part):
            scores = [{'name': key,
                    'scaled': float(self.scaled_scores[key][i].item()),  # Convert tensor to float
                    'unscaled': float(self.unscaled_scores[key][i].item()),
                    'desirability': float(self.desirability_scores[key][i].item())}
                    for key in self.unscaled_scores.keys()]
            particles.append({
                "smiles": self.smiles[i],
                "scores": scores,
                "dscore": self.fitness[i].item(),  # Convert tensor to float
                "v": self.v[i].round(3).tolist(),  # Round and convert to list
                "x": self.x[i].round(3).tolist(),  # Round and convert to list
                "part_best_x": self.particle_best_x[i].round(3).tolist(),  # Round and convert to list
                "part_best_fitness": self.particle_best_fitness[i].item()  # Convert tensor to float
            })

        output = {
            "particles": particles,
            "best_positions": [score.round(3).tolist() for score in self.history_swarm_best_x],  # Round and convert to list
            "best_fitness": self.swarm_best_fitness.item()  # Convert tensor to float
        }
        return output
    