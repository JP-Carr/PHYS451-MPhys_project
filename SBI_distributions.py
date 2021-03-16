from par_reader import pulsar_par_reader
from SBI_inference import observe
from q22_h0_conversion import q22_to_h0, h0_to_q22
from math import cos
import pickle
from sys import exit
import time
import multiprocessing as mp
import torch
import pandas as pd
from bilby.core.result import Result, ResultList
from bilby.core.prior import PriorDict, Uniform
from collections import OrderedDict
from cwinpy.hierarchical import MassQuadrupoleDistribution
from bilby.core.prior import HalfNormal
import numpy as np
from matplotlib import pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

num_pulsars=1000

parameter_dir="DAGout/test{}/pulsars".format(num_pulsars)
posterior_path="posteriors/posterior70000_SNPE.pkl"


phi0range = [0.0, np.pi]
psirange = [0.0, np.pi / 2.0]
cosiotarange = [-1.0, 1.0]
h0range = [0.0, 1e-23]
q22range=[0.0,1e38]

priors = OrderedDict()
priors["q22"] = Uniform(q22range[0], q22range[1], "q22", latex_label=r"$Q_{22}$"
)
priors["phi0"] = Uniform(
    phi0range[0], phi0range[1], "phi0", latex_label=r"$\\phi_0$", unit="rad"
)
priors["psi"] = Uniform(
    psirange[0], psirange[1], "psi", latex_label=r"$\psi$", unit="rad"
)
priors["cosiota"] = Uniform(
    cosiotarange[0], cosiotarange[1], "cosiota", latex_label=r"$\\cos{\\iota}$"
)
priors=PriorDict(priors)

def process(i):
    data=parameters.iloc[i]
    print("{} - {}   ".format(i,data["PSRJ"]), end="\r")
    h0=q22_to_h0(q22=float(data["Q22"]), dist=float(data["DIST"]), f0=float(data["F0"]))*1e25
    phi0=float(data["PHI0"])
    psi=float(data["PSI"])
    cosiota=cos(float(data["IOTA"]))
    
    f0=float(data["F0"])
    raj=data["RAJ"]
    decj=data["DECJ"]
    
    #print(h0)
    if h0<500:#1076:
        sample=observe(posterior, h0, phi0, psi, cosiota, F0=f0, RAJ=raj, DECJ=decj, plot=False, verbose=False, num_samples=10000)
        sample[:,0]=sample[:,0]/1e25
        sample[:,0]=h0_to_q22(sample[:,0], dist=float(data["DIST"]), f0=float(data["F0"]))
    else:
        sample=None
        
    return sample  
    
if __name__ == "__main__":
    start=time.time()

    try:    
        infile = open(posterior_path,'rb')       #Try to load relevent posterior 
        posterior = pickle.load(infile)
        infile.close()
        print("Prior Loaded - "+posterior_path)
        
    except FileNotFoundError:
        print(posterior_path+" not found")
        exit()
    try:
        parameters=pulsar_par_reader(parameter_dir)
    except FileNotFoundError:
        print(parameter_dir+" not found")
        exit()

    
    max_processes = mp.cpu_count()/2  # number of simultaneous processes cannot excede the number of logical processors
    #pool = mp.Pool(int(max_processes))
    with mp.Pool(int(max_processes)) as pool:
        processes = pool.map_async(process, range(len(parameters))) # Assign processes to the processing pool
        pool.close() #finish assigning
        pool.join() #begins multiprocessing
        raw_output=processes.get()
    
    num_processes=len(raw_output)
    output=[i for i in raw_output if i != None]
    successes=len(output)
    samples=output

    print("{} processes complete. {} outside of prior support".format(successes,num_processes-successes))
    print("\nSampling Runtime = {}s".format(round(time.time()-start,2)))
    

    df = pd.DataFrame(data=samples[0], columns=["q22", "phi0", "cosiota","psi"], dtype=float)

    res = Result(posterior=df, priors=priors, search_parameter_keys=list(priors.keys()))  # create a bilby result objects from the DataFrame

    reslist = ResultList([res])  # create a list of results

    sigma = 1e34# set half-normal prior on mean of exponential distribution
    distkwargs = {"mu": HalfNormal(sigma, name="mu")}
    distribution = "exponential"
    # set sampler parameters
 
    sampler_kwargs = {
        "sample": "unif",
        "nlive": 500,
        "gzip": True,
        "outdir": "exponential_distribution",
        "label": "test",
        "check_point_plot": False,
       # "sample": "rslice",
    }
    bins = 500
 
    # set MassQuadrupoleDistribution
    mqd = MassQuadrupoleDistribution(
        data=reslist,
        distribution="exponential",
        distkwargs=distkwargs,
        bins=bins,
        sampler_kwargs=sampler_kwargs,
    )

    count=0
    start2=time.time()
  
    a=plt.figure(1)
    plt.xlabel(r"$Q_{22}$")
    
    for sample in samples:
        q22=sample[:,0]
        y=np.arange(len(q22))
      #  print(max(q22))
     #   plt.plot(q22,y)
        plt.hist(q22, bins=100, alpha=0.5)
        
        if count==0:
            pass
        else:
            df = pd.DataFrame(data=sample, columns=["q22", "phi0", "cosiota","psi"], dtype=float)
            res = Result(posterior=df, priors=priors, search_parameter_keys=list(priors.keys()))            
            reslist = ResultList([res])
            mqd.add_data(reslist)
            print("{}/{}".format(count+1,len(samples)), end="\r")
        count+=1
      
    print("\nMQD Runtime = {}s".format(round(time.time()-start2,2)))
    print("\n")
    a.show()
  #  exit() 
    # run the sampler
    res = mqd.sample()
    # plot the samples
    hist=plt.figure(2)
    plt.xlabel("mu")
    plt.hist(res.posterior["mu"], bins=100)
    plt.axvline(x=sigma, color="r")
    print(len(mqd._posterior_samples))
    hist.show()