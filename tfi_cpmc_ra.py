import numpy as np
from scipy.linalg import expm
import math
import opt_einsum as oe
import copy

from tfi_utils import sigma_x, sigma_z

class ReanchoringCPMC:
    def __init__(
        self, 
        model, 
        phi_trial_init, 
        num_walkers, 
        len_episodes, 
        num_episodes, 
        num_episodes_sketch, 
        tt_rank, 
        sketch_rank, 
        dt, 
        num_episodes_sampling = 4, 
        gamma = 1.0, 
        alpha = 0.1
    ) -> None:

        self.model = model                                      # the Transverse-Field Ising model, an object of class TransverseFieldIsing
        self.phi_trial_init = phi_trial_init                    # the initial trial wavefunction in a TT form
                                                                # each core should be of shape (r_left, 2, r_right), 
                                                                # except possibly the first and last cores, 
                                                                # in practice we set r_left = r_right = 1 
        self.num_walkers = num_walkers                          # number of walkers
        self.len_episodes = len_episodes                        # number of steps in each episode
        self.num_episodes = num_episodes                        # number of episodes for CPMC in total
        self.num_episodes_sketch = num_episodes_sketch          # number of episodes for CPMC with TT-sketching to update the trial function
        self.tt_rank = tt_rank                                  # the targted TT-rank, an array of size dim - 1 (not including r_0 or r_d)
        self.sketch_rank = sketch_rank                          # the rank of the sketching tensor
        self.dt = dt                                            # imaginary-time step
        self.num_episodes_sampling = num_episodes_sampling      # number of bunches of walkers for doing sketching (time-averaging)
        self.gamma = gamma                                      # the discounting factor: walkers long ago should be as or less important
        self.alpha = alpha                                      # for the purpose of capturing low clusters for sketching


        self.num_spins = model.num_spins                 
        self.magnetic_field = model.magnetic_field

        self.bond_set = model.get_bond_set()                                 

        # ensure boundary TT cores of phi_trial_init have shape (1, 2, r) or (r, 2, 1)
        if len(phi_trial_init[0].shape) == 2:
            self.phi_trial_init[0] = self.phi_trial_init[0][np.newaxis, :, :]
        if len(phi_trial_init[-1].shape) == 2:
            self.phi_trial_init[-1] = self.phi_trial_init[-1][:, :, np.newaxis]

        self.b_halfk = expm(0.5 * dt * self.magnetic_field * sigma_x)    # half one-body propagator
        self.b_k = expm(dt * self.magnetic_field * sigma_x)              # one-body propagator

        # two-body propagators
        lmd = np.arccosh(np.exp(2 * dt)) / 2
        self.b_v = np.empty([2,2,2])                                     
        self.b_v[0] = expm(lmd * sigma_z)
        self.b_v[1] = expm(-lmd * sigma_z)

        # to ensure low clusters be considered for TT-sketching
        self.cluster_basis = np.array([[1,1],[alpha,-alpha]])


    def importance_function(self, random_walkers, phi_trial):
        """
        Calculate the importance function of all the random walkers
        i.e. max(<walker, phi_t>, 0). 

        Parameters
        random_walkers: list of numpy arrays
                shape of each array: (num_walkers, r_left, 2, r_right)

        Returns
        overlaps: numpy array, non-negative 
        """
        d = 0
        overlaps = np.einsum('npir,qis->nrs', random_walkers[d], phi_trial[d])
        for d in np.arange(1, self.num_spins - 1):
            temp = np.einsum('npir,qis->npqrs', random_walkers[d], phi_trial[d])
            overlaps = np.einsum('npq,npqrs->nrs', overlaps, temp)
        d = self.num_spins - 1
        temp = np.einsum('npir,qis->npq', random_walkers[d], phi_trial[d])
        overlaps = np.einsum('npq,npq->n', overlaps, temp)
        overlaps = np.maximum(overlaps, 0)

        return overlaps
    
    def one_body_prop(self, weights, overlaps, random_walkers, phi_trial, step):
        """
        Apply one-body propagator and update the walkers. 

        Parameters
        step: either 'full' or 'half'
        """
        if step == 'full':
            for d in range(self.num_spins):
                random_walkers[d] = np.einsum('ij,npjr->npir', self.b_k, random_walkers[d])
        if step == 'half':
            for d in range(self.num_spins):
                random_walkers[d] = np.einsum('ij,npjr->npir', self.b_halfk, random_walkers[d])

        overlaps_new = self.importance_function(random_walkers, phi_trial)
        for k in range(self.num_walkers):
            if weights[k] > 0:
                weights[k] = weights[k] * overlaps_new[k] / overlaps[k]
                overlaps[k] = overlaps_new[k]

        return weights, overlaps, random_walkers
    
    def update_importance_function_temp(self, random_walkers, phi_trial, i, j):
        """
        This is a fast implementation of update the importance function after a two-body propagator. 
        The key observation is that applying a two-body propagator only changes two sites of the random walkers, 
        thus we don't need to calculate the importance function all over again. 
        Instead, we calculate the overlap for other sites once, store them, and calculate the updated sites. 

        In this function, we calculate the overlap of those unchanged sites, and store them. 
        """
        if i > 0:
            d = 0
            v = np.einsum('npir,qis->nrs', random_walkers[d], phi_trial[d])
            for d in np.arange(1,i):
                temp = np.einsum('npir,qis->npqrs', random_walkers[d], phi_trial[d])
                v = np.einsum('npq,npqrs->nrs', v, temp)
            temp_left = v.copy()
        else:
            temp_left = None

        if j < self.num_spins - 1:
            d = self.num_spins - 1
            v = np.einsum('npir,qis->npq', random_walkers[d], phi_trial[d])
            for d in np.arange(j + 1, self.num_spins - 1)[::-1]:
                temp = np.einsum('npir,qis->npqrs', random_walkers[d], phi_trial[d])
                v = np.einsum('npqrs,nrs->npq', temp, v)
            temp_right = v.copy()
        else:
            temp_right = None

        if j - i > 1:
            d = i + 1
            v = np.einsum('npir,qis->npqrs', random_walkers[d], phi_trial[d])
            for d in np.arange(i+2,j):
                temp = np.einsum('npir,qis->npqrs', random_walkers[d], phi_trial[d])
                v = np.einsum('npqrs,nrstu->npqtu', v, temp)
            temp_middle = v.copy()
        else:
            temp_middle = None

        overlaps_temp = [temp_left, temp_middle, temp_right]

        return overlaps_temp
    
    def update_importance_function(self, random_walkers_new, phi_trial, i, j, overlaps_temp):
        """
        In this function, we use the pre-calculated overlap of unchanged sites, 
        then add the two updated sites to them. 
        """
        temp_left, temp_middle, temp_right = overlaps_temp

        if temp_left is not None:
            temp_i = np.einsum('npir,qis->npqrs', random_walkers_new[i], phi_trial[i])
            v_left = np.einsum('npq,npqrs->nrs', temp_left, temp_i)
        else:
            v_left = np.einsum('npir,qis->nrs', random_walkers_new[i], phi_trial[i])

        if temp_right is not None:
            temp_j = np.einsum('npir,qis->npqrs', random_walkers_new[j], phi_trial[j])
            v_right = np.einsum('npqrs,nrs->npq', temp_j, temp_right)
        else:
            v_right = np.einsum('npir,qis->npq', random_walkers_new[j], phi_trial[j])

        if temp_middle is not None:
            overlaps = oe.contract('npq,npqrs,nrs->n', v_left, temp_middle, v_right)
        else:
            overlaps = np.einsum('npq,npq->n', v_left, v_right)

        overlaps = np.maximum(overlaps, 0)

        return overlaps
    
    def propagate_an_episode(self, weights, overlaps, random_walkers, phi_trial):
        """
        Propagate an epsiode of CPMC, totally len_episodes steps. 
        In principle, we propagate b_halfk b_v b_halfk at each step, but in the middle steps, 
        we can combine the two b_halfk together. 
        Only at the first and the last step, we need to calculate b_halfk explicitly. 
        """

        # apply the half one-body propagator
        weights, overlaps, random_walkers = self.one_body_prop(weights, overlaps, random_walkers, phi_trial, step='half')
        weights = weights * self.num_walkers / sum(weights)

        # propagate len_episodes steps in one episode
        for t in range(self.len_episodes):

            # apply the two-body propagator, individually for each bond
            for bond in self.bond_set:
                random_walkers_p = copy.deepcopy(random_walkers)
                random_walkers_m = copy.deepcopy(random_walkers)
                i = min(bond) - 1
                j = max(bond) - 1

                random_walkers_p[i] = np.einsum('ij,npjr->npir', self.b_v[0], random_walkers[i])
                random_walkers_p[j] = np.einsum('ij,npjr->npir', self.b_v[0], random_walkers[j])
                random_walkers_m[i] = np.einsum('ij,npjr->npir', self.b_v[1], random_walkers[i])
                random_walkers_m[j] = np.einsum('ij,npjr->npir', self.b_v[1], random_walkers[j])
                
                # calculate the importance sampling probability
                overlaps_temp = self.update_importance_function_temp(random_walkers, phi_trial, i, j)
                overlaps_p = self.update_importance_function(random_walkers_p, phi_trial, i, j, overlaps_temp)
                overlaps_m = self.update_importance_function(random_walkers_m, phi_trial, i, j, overlaps_temp)

                sum_prob = overlaps_p + overlaps_m    
                for k in range(self.num_walkers):
                    if weights[k] > 0:
                        if sum_prob[k] == 0:
                            weights[k] = 0

                        else:
                            prob = np.array([overlaps_p[k], overlaps_m[k]]) / sum_prob[k]
                            x = np.random.choice([1,-1],p=prob)
                            if x == 1:
                                random_walkers[i][k] = copy.deepcopy(random_walkers_p[i][k])
                                random_walkers[j][k] = copy.deepcopy(random_walkers_p[j][k])
                            
                            else:
                                random_walkers[i][k] = copy.deepcopy(random_walkers_m[i][k])
                                random_walkers[j][k] = copy.deepcopy(random_walkers_m[j][k])

                            weights[k] = weights[k] * sum_prob[k] * 0.5 / overlaps[k]

                # update the importance function after applying a two-body propagator for each bond
                overlaps = self.update_importance_function(random_walkers, phi_trial, i, j, overlaps_temp)
                for k in range(self.num_walkers):
                    if overlaps[k] == 0:
                        weights[k] = 0

                weights = weights * self.num_walkers / sum(weights)    

            if t < self.len_episodes - 1:
                weights, overlaps, random_walkers = self.one_body_prop(weights, overlaps, random_walkers, phi_trial, step='full')
            else:
                weights, overlaps, random_walkers = self.one_body_prop(weights, overlaps, random_walkers, phi_trial, step='half')
                
            weights = weights * self.num_walkers / sum(weights)

        return weights, overlaps, random_walkers
    
    def population_control(self, weights, overlaps, random_walkers):
        random_walkers_adjusted = []
        for core in random_walkers:
            random_walkers_adjusted.append(core.copy())
        overlaps_new = np.zeros_like(overlaps)

        weights = weights * self.num_walkers / sum(weights) 
        sum_w = -np.random.uniform(0,1)
        duplicate_start = 0
        for i_walker in range(self.num_walkers):
            sum_w += weights[i_walker]
            n_duplicate = math.ceil(sum_w)
            for j_duplicate in np.arange(duplicate_start, n_duplicate):
                for d in range(self.num_spins):
                    walkers_copy = random_walkers[d][i_walker].copy()
                    random_walkers_adjusted[d][j_duplicate] = walkers_copy.copy()
                overlaps_new[j_duplicate] = overlaps[i_walker]
            duplicate_start = n_duplicate
            
        weights = np.ones(self.num_walkers)
        overlaps = overlaps_new.copy()
        
        return weights, overlaps, random_walkers_adjusted

    
    def measure_energy(self, weights, overlaps, random_walkers, h_phi_trial):
        d = 0
        local_energy = np.einsum('npir,mqis->nmrs', random_walkers[d], h_phi_trial[d])
        for d in np.arange(1, self.num_spins - 1):
            temp = np.einsum('npir,mqis->nmpqrs', random_walkers[d], h_phi_trial[d])
            local_energy = np.einsum('nmpq,nmpqrs->nmrs', local_energy, temp)
        d = self.num_spins - 1
        temp = np.einsum('npir,mqis->nmpq', random_walkers[d], h_phi_trial[d])
        local_energy = np.einsum('nmpq,nmpq->n', local_energy, temp)

        den = np.sum(weights)
        num = 0
        for k in range(len(weights)):
            if weights[k] > 0:
                num += weights[k] * local_energy[k] / overlaps[k]

        total_energy = num / den
        
        return total_energy

    def sketching(self, random_walkers_set):

        # squeeze r_0 = r_d = 1 for convenience

        random_walkers_set[0] = random_walkers_set[0][:,0,:,:]
        random_walkers_set[-1] = random_walkers_set[-1][:,:,:,0]

        # construct sketching tensors randomly
        left_sketch = []
        d = 0
        left_sketch.append(np.einsum('in,ij->jn',np.random.normal(0, 1, [2, self.sketch_rank]) / np.sqrt(self.sketch_rank), self.cluster_basis))
        for d in np.arange(1, self.num_spins - 1):
            left_sketch.append(np.einsum('min,ij->mjn',
                                         np.random.normal(0, 1, [self.sketch_rank, 2, self.sketch_rank]) / np.sqrt(self.sketch_rank), self.cluster_basis))
        d = self.num_spins - 1
        left_sketch.append(np.einsum('mi,ij->mj',np.random.normal(0, 1, [self.sketch_rank, 2]) / np.sqrt(self.sketch_rank), self.cluster_basis))

        right_sketch = []
        d = 0
        right_sketch.append(np.einsum('in,ij->jn',np.random.normal(0, 1, [2, self.sketch_rank]) / np.sqrt(self.sketch_rank), self.cluster_basis))
        for d in np.arange(1, self.num_spins - 1):
            right_sketch.append(np.einsum('min,ij->mjn',
                                         np.random.normal(0, 1, [self.sketch_rank, 2, self.sketch_rank]) / np.sqrt(self.sketch_rank), self.cluster_basis))
        d = self.num_spins - 1
        right_sketch.append(np.einsum('mi,ij->mj',np.random.normal(0, 1, [self.sketch_rank, 2]) / np.sqrt(self.sketch_rank), self.cluster_basis))

        # store the right sketching in prior

        right_cum_prod = []

        d = self.num_spins - 1
        core_prod = np.einsum('npi,qi->npq', random_walkers_set[d], right_sketch[d])
        cum_prod = core_prod.copy()
        right_cum_prod.append(cum_prod)
        for d in np.arange(1, self.num_spins - 1)[::-1]:
            core_prod = np.einsum('npir,qis->npqrs', random_walkers_set[d], right_sketch[d])
            cum_prod = np.einsum('npqrs,nrs->npq', core_prod, cum_prod)
            right_cum_prod.append(cum_prod)
        
        right_cum_prod = right_cum_prod[::-1]

        tt_cores = []

        # the first core

        d = 0

        bb = np.einsum('nir,nrs->is', random_walkers_set[d], right_cum_prod[d])
        core_temp = bb.copy()

        aa = np.einsum('ir,is->rs', left_sketch[d], core_temp)

        uu, ss, vv = np.linalg.svd(aa)
        core = np.dot(core_temp, vv[:self.tt_rank[d]].T)
        tt_cores.append(core / np.linalg.norm(core))

        aa_trimmed = np.dot(uu[:, :self.tt_rank[d]], np.diag(ss[:self.tt_rank[d]]))
        qq = np.dot(np.diag(1 / ss[:self.tt_rank[d]]), uu[:, :self.tt_rank[d]].T)

        core_prod = np.einsum('ir,nis->nrs', left_sketch[d], random_walkers_set[d])
        cum_prod = core_prod.copy()

        # the d-2 cores in the middle

        for d in np.arange(1, self.num_spins - 1):
            bb = np.einsum('nrs,nsit->nrit', cum_prod, random_walkers_set[d])
            bb = np.einsum('nrit,ntq->riq', bb, right_cum_prod[d])

            core_temp = np.einsum('pq,qir->pir', qq, bb)

            core_sketch = np.einsum('pir,qis->pqrs', left_sketch[d], core_temp)
            aa = np.einsum('pq,pqrs->rs', aa_trimmed, core_sketch)
            
            uu, ss, vv = np.linalg.svd(aa)
            core = np.einsum('piq,rq->pir', core_temp, vv[:self.tt_rank[d]])
            tt_cores.append(core / np.linalg.norm(core))

            aa_trimmed = np.dot(uu[:, :self.tt_rank[d]], np.diag(ss[:self.tt_rank[d]]))
            qq = np.dot(np.diag(1 / ss[:self.tt_rank[d]]), uu[:, :self.tt_rank[d]].T)

            core_prod = np.einsum('pir,nqis->npqrs', left_sketch[d], random_walkers_set[d])
            cum_prod = np.einsum('npq,npqrs->nrs', cum_prod, core_prod)

        # the last core

        d = self.num_spins - 1

        bb = np.einsum('npq,nqi->pi', cum_prod, random_walkers_set[d])

        core = np.dot(qq,bb)
        
        tt_cores.append(core / np.linalg.norm(core))

        # ensure boundary TT cores have shape (1, 2, r) or (r, 2, 1)

        tt_cores[0] = tt_cores[0][np.newaxis,:,:]
        tt_cores[-1] = tt_cores[-1][:,:,np.newaxis]

        return tt_cores
    
    def inner_prod(self, tt_1, tt_2):
        d = 0
        temp = np.einsum('pir,qis->rs', tt_1[d], tt_2[d])
        for d in np.arange(1, self.num_spins - 1):
            mat = np.einsum('pir,qis->pqrs', tt_1[d], tt_2[d])
            temp = np.einsum('pq,pqrs->rs', temp, mat)
        d = self.num_spins - 1
        mat = np.einsum('pir,qis->pq', tt_1[d], tt_2[d])
        temp = (temp * mat).sum()

        return temp
    
    def normalize(self, tt_cores):
        """
        Normalizing a TT so that it's norm is 1. 
        """
        tt_cores_norm_ave = np.sqrt(self.inner_prod(tt_cores, tt_cores)) ** (1 / self.num_spins)
        for d in range(self.num_spins):
            tt_cores[d] /= tt_cores_norm_ave

        return tt_cores

    def run(self):
        phi_trial = self.phi_trial_init
        h_phi_trial = self.model.get_h_phi(phi_trial)

        random_walkers = []
        for d in range(self.num_spins):
            random_walkers.append(np.tile(phi_trial[d], (self.num_walkers,1,1,1)))
        weights = np.ones(self.num_walkers)
        overlaps = self.importance_function(random_walkers, phi_trial)

        energy = self.measure_energy(weights, overlaps, random_walkers, h_phi_trial)
        energy_measurements = [energy]
        print('Episodes:', 0, 'Energy:', energy)

        random_walkers_set = copy.deepcopy(random_walkers)
        num_walkers_sketch = [self.num_walkers]

        for episode in 1 + np.arange(self.num_episodes):
            weights, overlaps, random_walkers = self.propagate_an_episode(weights, overlaps, random_walkers, phi_trial)

            energy = self.measure_energy(weights, overlaps, random_walkers, h_phi_trial)
            energy_measurements.append(energy)
            print('Episodes:', episode, 'Energy:', energy)

            if episode <= self.num_episodes_sketch:
                non_zero_idx = []
                ratio = []
                for k in range(self.num_walkers):
                    if weights[k] > 0:
                        non_zero_idx.append(k)
                        ratio.append(weights[k] / overlaps[k])
                random_walkers_current = []
                for d in range(self.num_spins):
                    random_walkers_current.append(random_walkers[d][non_zero_idx])
                random_walkers_current[0] = np.einsum('n,npir->npir', np.array(ratio), random_walkers_current[0])
                nn = len(non_zero_idx)

                if episode < self.num_episodes_sampling:
                    d = 0
                    random_walkers_set[d] = np.vstack([self.gamma * random_walkers_set[d], random_walkers_current[d]])
                    for d in np.arange(1, self.num_spins):
                        random_walkers_set[d] = np.vstack([random_walkers_set[d], random_walkers_current[d]])
                    num_walkers_sketch.append(nn)

                else:
                    d = 0
                    random_walkers_set[d] = np.vstack([self.gamma * random_walkers_set[d][num_walkers_sketch[0]:], random_walkers_current[d]])
                    for d in np.arange(1, self.num_spins):
                        random_walkers_set[d] = np.vstack([random_walkers_set[d][num_walkers_sketch[0]:], random_walkers_current[d]])
                    del num_walkers_sketch[0]
                    num_walkers_sketch.append(nn)

                random_walkers_set_copy = copy.deepcopy(random_walkers_set)
                phi_trial = self.sketching(random_walkers_set_copy)

                phi_trial = self.normalize(phi_trial)
                h_phi_trial = self.model.get_h_phi(phi_trial)

                # update the importance function, since the trial function has changed

                overlaps_new = self.importance_function(random_walkers, phi_trial)
                for k in range(self.num_walkers):
                    if weights[k] > 0:
                        weights[k] = weights[k] * overlaps_new[k] / overlaps[k]
                        overlaps[k] = overlaps_new[k]


            weights, overlaps, random_walkers = self.population_control(weights, overlaps, random_walkers)

        return energy_measurements