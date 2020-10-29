from bilby.core.prior import PriorDict, Uniform
import numpy as np
import torch
from torch.distributions.uniform import Uniform as torch_uni

class PriorDictMod(PriorDict):
    pass

    def log_prob(self, sample):
        return self.ln_prob(sample)
    
    def array_convertion(self):
        return np.array([self[i] for i in self])
    
#testing
if __name__=="__main__":
    """
    prior = {}
    prior["h0"] = Uniform(0.0, 1e-22, "h0")
    prior["phi0"] = Uniform(0.0, np.pi, "phi0")
    priordic = PriorDictMod(prior)
    sam=priordic.sample()
    print(sam)
    print(priordic.log_prob(sam))
    """
    
    
class distribution_array:
    def __init__(self, dist_array):
        self.dist_array=dist_array
        
    def sample(self, sample_shape=()): #returns a tensor of samples
   
        samples=[i.sample(sample_shape) for i in self.dist_array]
        
        print(samples)
        x=0
        return x
    
    
    def log_prob(self, sample): #needs to produce a single value
        return 

if __name__=="__main__":
    shape=(2,)
    
    a=torch.tensor([12,8])
    b=torch.tensor([7,89])
    c=torch.cat((a,b),0)
  #  print(c)
    
   # print(torch.tensor([12,8],[7,89]))
    #print(1/0)
    h0=torch_uni(0.0, 1e-22)
    phi0=torch_uni(0.0, np.pi)
    ar=distribution_array(np.array([h0,phi0]))
    print(ar.sample(shape))
