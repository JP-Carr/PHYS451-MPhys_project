import torch
import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np
import sys

from bilby.core.prior import PriorDict, Uniform
from PriorDictMod import PriorDictMod, distribution_array
from torch.distributions.uniform import Uniform as torch_uni

import sbi.utils as utils

h0=torch_uni(0.0, 1e-22)
phi0=torch_uni(0.0, np.pi)

dist_vals={"h0": [0.0, 1e-22],
           "phi0": [0.0, np.pi]
           }

dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])
dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])

prior = utils.BoxUniform(low=dist_lows, high=dist_highs)
print(prior)



def simulator(parameter_set):   #links parameters to simulation data
    print(parameter_set.shape)

    return parameter_set

posterior = infer(simulator, prior, method='SNPE', num_simulations=100)

"""
observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))

print("\a")
"""