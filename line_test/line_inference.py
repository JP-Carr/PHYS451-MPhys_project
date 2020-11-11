import torch
import sbi.utils as utils

from sbi.inference import SNPE, prepare_for_sbi, SNLE, SNRE

import time

import pickle
from multiprocessing import cpu_count
import subprocess as sp
import os
import sbi.utils as utils

sim_iterations=10000 #3 minimum
sim_method="SNPE"   #SNPE, SNLE, SNRE
use_CUDA=False
observe=True
save_posterior=False
shutdown=False

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
    
    dist_vals={"m": [-100., 100.],    #parameter distributions [low, highs]
               "c": [-100., 100.]
               }
    
    dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])
    dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])
    
    prior = utils.BoxUniform(low=dist_lows, high=dist_highs)
  #  print(prior.sample())
    
 #   sim_counter=-1  # 2 runs occur during setup
 
    def line(m,c=0.):
        x=torch.arange(-100.,100.,0.1)
        return m*x+c
 
    def simulator(parameter_set):   #links parameters to simulation data
      #  global sim_timer, sim_counter
       
     #   sim_counter+=1
    #    startx=time.time()
        

        m=float(parameter_set[0])
        c=float(parameter_set[1])

   #     sim_timer.append(time.time()-startx)
    
        if use_CUDA==True:
            get_gpu_memory()
            
        return line(m,c)#parameter_set
    
    
    
    try:
        threads=cpu_count()
    except:
        threads=1
    
    simulator, prior = prepare_for_sbi(simulator, prior) 
    
    if sim_method=="SNPE":
        inference = SNPE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif sim_method=="SNLE":
        inference = SNLE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif sim_method=="SNRE":
        inference = SNRE(simulator, prior, num_workers=threads, device=device)
    
    posterior = inference(num_simulations=sim_iterations, proposal=None)

    print("\nTraining Duration = {}s".format(round(time.time()-start,2)))
#    print("Total Simulation Time = {}s".format(round(sum(sim_timer),2)))
    
    if save_posterior==True:
        pickler(posterior_path,posterior)

if observe==True:
    observation=line(15.2,32.5)
 #   print(observation.size())
 #   observation=torch.zeros(1440)
    print(observation)
    samples = posterior.sample((10000,), x=observation)#,sample_with_mcmc=True)
    #print(samples)
   # print("-----------------------------------------")
    log_probability = posterior.log_prob(samples, x=observation,norm_posterior=False)
    #print(len(log_probability))
    
    labels=[i for i in dist_vals]
    _ = utils.pairplot(samples, limits=None, fig_size=(6,6), labels=labels)
print("\a")

if shutdown==True:
    time.sleep(60)
    os.system("shutdown") 