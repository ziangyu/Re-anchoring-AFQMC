import numpy as np

from tfi_cpmc_ra import ReanchoringCPMC
from tfi_model import TransverseFieldIsing
from tfi_utils import disordered_state

# model parameters

num_spins_x = 4
num_spins_y = 4
boundary_x = 'p'   # periodic boundary, 'o' for open
boundary_y = 'p'   # periodic boundary, 'o' for open
magnetic_field = 3

num_spins = num_spins_x * num_spins_y

# CPMC with re-anchoring parameters

dt = 0.01
num_walkers = 2000
len_episodes = 50
num_episodes = 100
num_episodes_sketch = 40

tt_rank = np.ones(num_spins - 1, dtype=int) * 4
tt_rank[0], tt_rank[-1] = 2, 2

sketch_rank = 60

model = TransverseFieldIsing(magnetic_field, num_spins_x, num_spins_y)
cpmc_solver = ReanchoringCPMC(model, disordered_state(num_spins), num_walkers, len_episodes, 
                              num_episodes, num_episodes_sketch, tt_rank, sketch_rank, dt)

energy_measurements = cpmc_solver.run()