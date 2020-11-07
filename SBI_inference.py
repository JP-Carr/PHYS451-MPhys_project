import torch
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi
import numpy as np
import sys
import time
from bilby.core.prior import PriorDict, Uniform
from PriorDictMod import PriorDictMod, distribution_array
from torch.distributions.uniform import Uniform as torch_uni
from HeterodynedData import generate_het
import pickle
from multiprocessing import cpu_count

import subprocess as sp
import os

sim_iterations=5000 #3 minimum
sim_method="SNLE"
use_CUDA=False


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  #ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

def pickler(path,obj):
    outfile = open(path,'wb')
    pickle.dump(obj,outfile)
    outfile.close()
    print(path+" pickled")

if use_CUDA==True:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device="gpu"
else:
    device="cpu"


posterior_path="posteriors/posterior{}_{}.pkl".format(sim_iterations,sim_method)
try:    
    infile = open(posterior_path,'rb')
    posterior = pickle.load(infile)
    infile.close()
    print("Prior Loaded - "+posterior_path)
    
except FileNotFoundError:
    
    print(posterior_path+" not found.\nGenerating posterior")

    start=time.time()
    sim_timer=[]
    
    dist_vals={"h0": [0.0, 1e-22]    #parameter distributions [low, highs]
               #"phi0": [0.0, np.pi]
               }
    
    dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])
    dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])
    
    prior = utils.BoxUniform(low=dist_lows, high=dist_highs)
  #  print(prior.sample())
    
    
    sim_counter=-1  # 2 runs occur during setup
    def simulator(parameter_set):   #links parameters to simulation data
        global sim_timer, sim_counter
        
      #  if sim_counter>0:
          #  sys.stdout.write("Performing Simulations: {}/{}    ".format(sim_counter,sim_iterations))
         #   sys.stdout.flush()
           # print("Performing Simulations: {}/{}   ".format(sim_counter,sim_iterations) , end="\r", flush=True)
        
        sim_counter+=1
        startx=time.time()
        H0=float(parameter_set)#[0])
        #phi0=float(parameter_set[1])
        
        het=generate_het(H0=H0)#,PHI0=phi0)
        sim_timer.append(time.time()-startx)
    
        if use_CUDA==True:
            get_gpu_memory()

        return torch.from_numpy(het.data)#parameter_set
    
    try:
        threads=cpu_count()
    except:
        threads=1
 #   print(threads)
    
    #posterior = infer(simulator, prior, method=sim_method, num_simulations=sim_iterations, num_workers=threads)
    
    simulator, prior = prepare_for_sbi(simulator, prior)   
    inference = SNPE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    posterior = inference(num_simulations=sim_iterations, proposal=None)

    print("\nTraining Duration = {}s".format(round(time.time()-start,2)))
    print("Total Simulation Time = {}s".format(round(sum(sim_timer),2)))
    
    pickler(posterior_path,posterior)
"""
observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))
"""
#sys.exit()
observation=torch.from_numpy(generate_het(H0=-5.12e-25).data)
print(observation)
samples = posterior.sample((10000,), x=observation)
print("-----------------------------------------")
log_probability = posterior.log_prob(samples, x=observation)
print(log_probability)
print("\a")