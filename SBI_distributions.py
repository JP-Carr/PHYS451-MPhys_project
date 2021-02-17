from par_reader import pulsar_par_reader
from SBI_inference import observe
from q22_h0_conversion import q22_to_h0, h0_to_q22
from math import cos
import pickle
from sys import exit
import time
import multiprocessing as mp

parameter_dir="DAGout/test/pulsars"
posterior_path="posteriors/posterior70000_SNPE.pkl"

def process(i):
    print(i)
    data=parameters.iloc[i]
    
    h0=q22_to_h0(q22=float(data["Q22"]), dist=float(data["DIST"]), f0=float(data["F0"]))*1e25
    phi0=float(data["PHI0"])
    psi=float(data["PSI"])
    cosiota=cos(float(data["IOTA"]))
    #print(h0)
    sample=observe(posterior, h0, phi0, psi, cosiota, plot=False)
    sample[:,0]=sample[:,0]/1e25
    return sample

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
for i in range(len(parameters)):
    data=parameters.iloc[i]
    
    h0=q22_to_h0(q22=float(data["Q22"]), dist=float(data["DIST"]), f0=float(data["F0"]))
    phi0=float(data["PHI0"])
    psi=float(data["PSI"])
    cosiota=cos(float(data["IOTA"]))
    #print(h0)
    sample=observe(posterior, h0*1e25, phi0, psi, cosiota, plot=False)
    sample[:,0]=sample[:,0]/1e25
    print(sample)
"""   
   
if __name__ == "__main__":
    max_processes = mp.cpu_count()  # number of simultaneous processes cannot excede the number of logical processors
    pool = mp.Pool(max_processes)
    processes = pool.map_async(process, range(len(parameters))) # Assign processes to the processing pool
    pool.close() #finish assigning
    pool.join() #begins multiprocessing
    output=processes.get()
    print(len(output))
    print("\nRuntime = {}s".format(round(time.time()-start,2)))
    time.sleep(20)