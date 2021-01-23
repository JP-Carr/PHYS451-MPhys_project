import torch
import sbi.utils as utils
from sbi.inference import SNPE, prepare_for_sbi, SNLE, SNRE
import numpy as np
import time
from HeterodynedData import generate_het
import pickle
from multiprocessing import cpu_count
import subprocess as sp
import os
from matplotlib import pyplot as plt

#VARIABLES---------------------------------------------------------------------

sim_iterations=50000  # Number of simulation to be performed during posterior generation(3 minimum)
inf_method="SNPE"    # SBI inference method (SNPE, SNLE, SNRE)
use_CUDA=False       # Utilise GPU during training - not recommended
observe=True        # Perform parameter estimation on test GW?
save_posterior=True  # Save generated posterior?
shutdown=False       # Shutdown device after script completion?

observation_parameters={r"$H_0\times 10^{25}$": 5.12e-23 *1e25,   # paramters for test GW (must be floats)
                        r"$\phi_0$": 2.8,
                        r"$cos(\iota)$": 0.3,
                        r"$\psi$": 0.82
                        }

dist_vals={r"$H_0\times 10^{25}$": torch.tensor([0., 1e-22]) *1e25,    #parameter distributions [low, highs]
               r"$\phi_0$": [0., np.pi],
               r"$cos(\iota)$": [-1., 1.],
               r"$\psi$": [0., np.pi/2]
               }

#FUNCTIONS---------------------------------------------------------------------

def get_gpu_memory():
    """
    Provides the unused VRAM in system GPUs (NVIDIA only)

    Returns
    -------
    memory_free_values : list[int]
        Unused VRAM in GPUs (MB).

    """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    #ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

def pickler(path,obj):
    """
    Pickles an object and saves it to a chosen location

    Parameters
    ----------
    path : str
        Path of directory to save .pck file.
    obj : Python object 
        Object to pickle.

    Returns
    -------
    None.

    """
    outfile = open(path,'wb')
    pickle.dump(obj,outfile)
    outfile.close()
    print(path+" pickled")
    
def simulator(parameter_set):   #links parameters to simulation data
    """
    Generated heterodyned data from provided parameters

    Parameters
    ----------
    parameter_set : torch.Tensor
        Path of directory to save .pck file.

    Returns
    -------
    het_data: numpy.array
        Complex time series of heterodyned data

    """
   # print(type(parameter_set))
    
    h0=float(parameter_set[0])
    phi0=float(parameter_set[1])
    cosiota=float(parameter_set[2])
    psi=float(parameter_set[3])
  #  print(h0,phi0,cosiota,psi)
    het=generate_het(H0=h0, PHI0=phi0, COSIOTA=cosiota, PSI=psi).data

    if use_CUDA==True:
        get_gpu_memory()
        
    r=het.real    
    i=het.imag    
    c=np.concatenate((r,i))
    het_data=torch.from_numpy(c)#parameter_set
    return het_data
    
#SCRIPT------------------------------------------------------------------------    
    

if use_CUDA==True:  #Set up device to be used for inference
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device="gpu"
else:
    device="cpu"
    
try:
    threads=cpu_count()  # detirmines number of workers to be used during posterior generation
except:
    threads=1


posterior_path="posteriors/posterior{}_{}.pkl".format(sim_iterations,inf_method) # Posterior save location

try:    
    infile = open(posterior_path,'rb')       #Try to load relevent posterior 
    posterior = pickle.load(infile)
    infile.close()
    print("Prior Loaded - "+posterior_path)

except FileNotFoundError:
    
    print(posterior_path+" not found.\nGenerating posterior")

    start=time.time()

    
    dist_lows=torch.tensor([float(dist_vals[i][0]) for i in dist_vals])   
    dist_highs=torch.tensor([float(dist_vals[i][1]) for i in dist_vals])
    
    prior = utils.BoxUniform(low=dist_lows, high=dist_highs)   # prior constucted from parameter ranges
    
    simulator, prior = prepare_for_sbi(simulator, prior) 
    
    if inf_method=="SNPE":          #Set inference method
        inference = SNPE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif inf_method=="SNLE":
        inference = SNLE(simulator, prior, density_estimator='mdn', num_workers=threads, device=device)
    elif inf_method=="SNRE":
        inference = SNRE(simulator, prior, num_workers=threads, device=device)
    
    posterior = inference(num_simulations=sim_iterations, proposal=None) # generate prior

    print("\nTraining Duration = {}s".format(round(time.time()-start,2)))
    
    if save_posterior==True:
            pickler(posterior_path,posterior)


if observe==True:
    ob_het=generate_het(H0=observation_parameters[r"$H_0\times 10^{25}$"], PHI0=observation_parameters[r"$\phi_0$"],PSI=observation_parameters[r"$\psi$"] , COSIOTA=observation_parameters[r"$cos(\iota)$"]).data
    observation=torch.from_numpy(np.concatenate((ob_het.real,ob_het.imag)))
    samples = posterior.sample((50000,), x=observation)
    
    log_probability = posterior.log_prob(samples, x=observation,norm_posterior=False)

    labels=[i for i in observation_parameters]
    points=np.array([observation_parameters[i] for i in observation_parameters])

    _ = utils.pairplot(samples, limits=None, fig_size=(6,6), labels=labels ,points=points)  # plot results
    print("\a")
    plt.show()
else:
    print("\a")
    
if shutdown==True:
    time.sleep(60)
    os.system("shutdown") 