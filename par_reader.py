from os import listdir
import linecache
import time
import torch 

target_dir="DAGout/test/pulsars"
pars=[i for i in listdir(target_dir) if i.endswith(".par")]
#print(pars)

start=time.time()

parameters=torch.zeros(len(pars),4)
count=0
for path in pars:
    try:
        data=[float(linecache.getline(target_dir+"/"+path, line).strip().split(" ")[-1]) for line in (6,7,8,9)] #order: phi,iota,phi0,q22
        parameters[count]=torch.tensor(data)
    except Exception as e:
        print(e)
       
    count+=1   
print(parameters)    

print("\nRuntime = {}s".format(round(time.time()-start,2)))