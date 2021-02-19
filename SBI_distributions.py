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

parameter_dir="DAGout/test/pulsars"
posterior_path="posteriors/posterior70000_SNPE.pkl"

def process(i):
    print(i)
    print()
    data=parameters.iloc[i]
    
    h0=q22_to_h0(q22=float(data["Q22"]), dist=float(data["DIST"]), f0=float(data["F0"]))*1e25
    phi0=float(data["PHI0"])
    psi=float(data["PSI"])
    cosiota=cos(float(data["IOTA"]))
    
    #print(h0)
    if h0<1076:
        sample=observe(posterior, h0, phi0, psi, cosiota, plot=False, verbose=False, num_samples=50000)
        sample[:,0]=sample[:,0]/1e25
        sample[:,0]=h0_to_q22(sample[:,0], dist=float(data["DIST"]), f0=float(data["F0"]))
    else:
        sample=None
    """
    x=open("values.txt","a")
    x.write(str(i)+",")
    x.close()
    """
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
        
    """    
    output=[]   
    print(len(parameters))
    for i in range(len(parameters)):
        sample=process(i)
      #  print(sample)
        output.append(sample)
    """ 
        
    """
    x=parameters.iloc[47]
    h0=q22_to_h0(q22=float(x["Q22"]), dist=float(x["DIST"]), f0=float(x["F0"]))*1e25
    phi0=float(x["PHI0"])
    psi=float(x["PSI"])
    cosiota=cos(float(x["IOTA"]))
    print(h0,phi0,cosiota,psi)
    exit()
    """
    
    max_processes = mp.cpu_count()/2  # number of simultaneous processes cannot excede the number of logical processors
    pool = mp.Pool(int(max_processes))
    processes = pool.map_async(process, range(len(parameters))) # Assign processes to the processing pool
    pool.close() #finish assigning
    pool.join() #begins multiprocessing
    raw_output=processes.get()
    
    num_processes=len(raw_output)
    output=[i for i in raw_output if i != None]
    successes=len(output)
    samples=torch.cat(output,0)
    #print(samples)
    print("{} processes complete. {} outside of prior support".format(successes,num_processes-successes))
    print("\nSampling Runtime = {}s".format(round(time.time()-start,2)))
    
    df = pd.DataFrame(data=samples, columns=["q22", "phi0", "cosiota","psi"])
    res = Result(posterior=df)  # create a bilby result objects from the DataFrame
    reslist = ResultList([res])  # create a list of results
  
    
    from cwinpy.hierarchical import MassQuadrupoleDistribution
    from bilby.core.prior import HalfNormal
    # set half-normal prior on mean of exponential distribution
    sigma = 1e34
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
    bins = 1000
    # set MassQuadrupoleDistribution
    mqd = MassQuadrupoleDistribution(
        data=reslist,
        distribution="exponential",
        distkwargs=distkwargs,
        bins=bins,
        sampler_kwargs=sampler_kwargs,
    )
    # run the sampler
    res = mqd.sample()
    # plot the samples
    from matplotlib import pyplot as pl
    pl.hist(res.posterior["mu"], bins=20)
    
    _=input("Press any key to exit\n")