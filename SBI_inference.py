import torch
import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np
import sys
import time
from bilby.core.prior import PriorDict, Uniform
from PriorDictMod import PriorDictMod, distribution_array
from torch.distributions.uniform import Uniform as torch_uni
from HeterodynedData import generate_het
import sbi.utils as utils

start=time.time()
sim_timer=[]
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


#sys.exit()

dist_vals={"h0": [0.0, 1e-22],
           "phi0": [0.0, np.pi]
           }

dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])
dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])

prior = utils.BoxUniform(low=dist_lows, high=dist_highs)
print(prior.sample())



def simulator(parameter_set):   #links parameters to simulation data
    global sim_timer
    start=time.time()
    H0=float(parameter_set[0])
    phi0=float(parameter_set[1])
    
    het=generate_het(H0=H0,PHI0=phi0)
    sim_timer.append(time.time()-start)
    return torch.from_numpy(het.data)#parameter_set

posterior = infer(simulator, prior, method='SNPE', num_simulations=4)

"""
observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))


"""
print("\nTime elapsed = {}s\a".format(round(time.time()-start,2)))
print("Total Simulation Time = {}s".format(round(sum(sim_timer),2)))