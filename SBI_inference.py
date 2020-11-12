import torch
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, SNLE, SNRE
import numpy as np
import sys
import time
from bilby.core.prior import PriorDict, Uniform
from torch.distributions.uniform import Uniform as torch_uni
from HeterodynedData import generate_het
import pickle
from multiprocessing import cpu_count
import subprocess as sp
import os

sim_iterations=50000 #3 minimum
sim_method="SNPE"   #SNPE, SNLE, SNRE
use_CUDA=False
observe=True
save_posterior=True
shutdown=False

observation_parameters={"H0*1e25": 5.12e-23 *1e25,
                        "phi0": 2.8,
                        "cosiota": 0.3,
                        "psi": 0.82
                        }

dist_vals={"H0*1e25": torch.tensor([0., 1e-22]) *1e25,    #parameter distributions [low, highs]
               "phi0": [0., np.pi],
               "cosiota": [-1., 1.],
               "psi": [0., np.pi/2]
               }

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
    
    
    
    dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])
    dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])
    
    prior = utils.BoxUniform(low=dist_lows, high=dist_highs)
  #  print(prior.sample())
    
    sim_counter=-1  # 2 runs occur during setup
    def simulator(parameter_set):   #links parameters to simulation data
        global sim_timer, sim_counter
        
       # if sim_counter>0:
          #  sys.stdout.write("Performing Simulations: {}/{}    ".format(sim_counter,sim_iterations))
         #   sys.stdout.flush()
        #    print("Performing Simulations: {}/{}   ".format(sim_counter,sim_iterations))# , end="\r", flush=True)
        
        sim_counter+=1
        startx=time.time()
        h0=float(parameter_set[0])
        phi0=float(parameter_set[1])
        cosiota=float(parameter_set[2])
        psi=float(parameter_set[3])
  #      print("H_0 = "+str(H0))
        het=generate_het(H0=h0, PHI0=phi0, COSIOTA=cosiota, PSI=psi)
        
        sim_timer.append(time.time()-startx)
    
        if use_CUDA==True:
            get_gpu_memory()
        #print(het.data)
        return torch.from_numpy(het.data)#parameter_set
    
    try:
        threads=cpu_count()
    except:
        threads=1
 #   print(threads)
    
    #posterior = infer(simulator, prior, method=sim_method, num_simulations=sim_iterations, num_workers=threads)
    
    simulator, prior = prepare_for_sbi(simulator, prior) 
    
    if sim_method=="SNPE":
        inference = SNPE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif sim_method=="SNLE":
        inference = SNLE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif sim_method=="SNRE":
        inference = SNRE(simulator, prior, num_workers=threads, device=device)
    
    posterior = inference(num_simulations=sim_iterations, proposal=None)

    print("\nTraining Duration = {}s".format(round(time.time()-start,2)))
    #print("Total Simulation Time = {}s".format(round(sum(sim_timer),2)))
    
    if save_posterior==True:
        pickler(posterior_path,posterior)
"""
observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
_ = utils.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], fig_size=(6,6))
"""
if observe==True:
    observation=torch.from_numpy(generate_het(H0=observation_parameters["H0*1e25"], PHI0=observation_parameters["phi0"], COSIOTA=observation_parameters["cosiota"]).data)
 #   print(observation.size())
 #   observation=torch.zeros(1440)
   # print(observation)
    samples = posterior.sample((10000,), x=observation)#,sample_with_mcmc=True)
  #  print(samples)
   # print("-----------------------------------------")
    log_probability = posterior.log_prob(samples, x=observation,norm_posterior=False)
   # print(log_probability)
    
    labels=[i for i in observation_parameters]
    points=np.array([observation_parameters[i] for i in observation_parameters])
   # print(points)
    _ = utils.pairplot(samples, limits=None, fig_size=(6,6), labels=labels ,points=points)
    
print("\a")

if shutdown==True:
    time.sleep(60)
    os.system("shutdown") 