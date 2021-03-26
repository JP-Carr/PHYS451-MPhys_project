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

sim_iterations=80000  # Number of simulation to be performed during posterior generation(3 minimum)
inf_method="SNPE"    # SBI inference method (SNPE, SNLE, SNRE)
use_CUDA=False       # Utilise GPU during training - not recommended
perform_observation=True   # Perform parameter estimation on test GW?
save_posterior=True  # Save generated posterior?
shutdown=False       # Shutdown device after script completion?

observation_parameters={r"$H_0\times 10^{25}$": 9.087957135017964e-26*1e25,#1.1e-23 *1e25,#5.12e-23 *1e25,   # paramters for test GW (must be floats)
                        r"$\phi_0$": 0.7769275287194411517,#2.8,
                        r"$cos(\iota)$": 0.515496,#0.3,
                        r"$\psi$": 1.4744513502625564705#0.82
                        }

dist_vals={r"$H_0\times 10^{25}$": torch.tensor([0., 1e-22]) *1e25,    #parameter distributions [low, highs]
               r"$\phi_0$": [0., np.pi],
               r"$cos(\iota)$": [-1., 1.],
               r"$\psi$": [0., np.pi/2]
               } #j1727-0755

SNR=5

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
    het=generate_het(H0=h0, PHI0=phi0, COSIOTA=cosiota, PSI=psi).data*1e25

    if use_CUDA==True:
        get_gpu_memory()
        
    r=het.real    
    i=het.imag    
    c=np.concatenate((r,i))
    het_data=torch.from_numpy(c)   #parameter_set
    return het_data
    
def observe(posterior, h0, phi0, psi, cosiota, F0=None, RAJ=None, DECJ=None, plot=True, num_samples=50000, verbose=False):
    start=time.time()
   # print(parameters[r"$H_0\times 10^{25}$"]/5)
    if F0==None or RAJ==None or DECJ==None:
        print("fallback")
        ob_het=generate_het(H0=float(h0), PHI0=float(phi0), PSI=float(psi), COSIOTA=float(cosiota)).data*1e25
    else:
        ob_het=generate_het(H0=float(h0), PHI0=float(phi0), PSI=float(psi), COSIOTA=float(cosiota), F0=F0, RAJ=RAJ, DECJ=DECJ).data*1e25
    observation=torch.from_numpy(np.concatenate((ob_het.real,ob_het.imag)))
    samples = posterior.sample((num_samples,), x=observation, show_progress_bars=verbose)

    if plot==True:
        two_sigma=np.percentile(samples, 95,axis=0)
        minus_two_sigma=np.percentile(samples, 100-95,axis=0)
        one_sigma=np.percentile(samples, 68,axis=0)
        minus_one_sigma=np.percentile(samples, 100-68,axis=0)
        
    
        labels=[i for i in observation_parameters]
        points=[[j for j in one_sigma],[k for k in minus_one_sigma],[l for l in two_sigma],[n for n in minus_two_sigma],[observation_parameters[i] for i in observation_parameters]]
       # print(points)
    
        colours=['#ff7f0e', '#ff7f0e',"#FF0000","#FF0000",'#1f77b4']
        plot = utils.pairplot(samples, limits=None, fig_size=(6,6), labels=labels ,points=points, points_colors=colours)  # plot results
        if verbose==True:
            print("\nInference Duration = {}s\a".format(round(time.time()-start,2)))
        plt.show()
    elif verbose==True:
        print("\nInference Duration = {}s\a".format(round(time.time()-start,2)))
 #   print("\a")
    return samples

#SCRIPT------------------------------------------------------------------------    
if __name__=="__main__":
    start=time.time()    
    
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

    
    if perform_observation==True:
     #   observe(posterior, h0=observation_parameters[r"$H_0\times 10^{25}$"], phi0=observation_parameters[r"$\phi_0$"], psi=observation_parameters[r"$\psi$"], cosiota=observation_parameters[r"$cos(\iota)$"], verbose=True)
        observe(posterior, h0=observation_parameters[r"$H_0\times 10^{25}$"], phi0=observation_parameters[r"$\phi_0$"], psi=observation_parameters[r"$\psi$"], cosiota=observation_parameters[r"$cos(\iota)$"], verbose=True, F0=282.2588318211538194191, RAJ="17:27:27.005928305039", DECJ="-07:55:08.752768153206")
    else:
        print("\a")
    
    if shutdown==True:
        time.sleep(60)
        os.system("shutdown") 