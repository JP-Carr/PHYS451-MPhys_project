#from sbi.examples.minimal import simple
#posterior = simple()
#print(posterior)

import torch
import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np

#print(0)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_dim = 3
x=2*torch.ones(num_dim)

prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))
#print(x.device)

print("-----------------")
def simulator(parameter_set):
    #print(parameter_set)
  #  print((1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1).device)
    return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1

posterior = infer(simulator, prior, method='SNPE', num_simulations=1000)
#x=p4sbi(simulator, prior)
#posterior = SNPE_C(x, device="gpu")

observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))

print("\a")