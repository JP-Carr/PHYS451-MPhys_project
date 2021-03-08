from os import listdir
import linecache
import time
import pandas as pd

def pulsar_par_reader(pulsar_dir):
    print("Processing {} pulsars...".format(len(listdir(pulsar_dir))))
    start=time.time()
    pars=[i for i in listdir(pulsar_dir) if i.endswith(".par")]

    parameters=[[linecache.getline(pulsar_dir+"/"+path, line).strip().split(" ")[-1] for line in range(1,10)] for path in pars]    
        
    labels=[linecache.getline(pulsar_dir+"/"+pars[0], line).strip().split(" ")[0] for line in range(1,10)]
    df=pd.DataFrame(data=parameters, columns=labels)
   # print(labels)
    #print("\nReader Runtime = {}s".format(round(time.time()-start,4)))
    return df



if __name__=="__main__":
    target_dir="DAGout/test/pulsars"
    df=pulsar_par_reader(target_dir)
    
    print(df["Q22"])