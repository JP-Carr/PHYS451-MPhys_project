from matplotlib import pyplot as plt
import pickle
import numpy as np
from sys import exit
pkl_path="/home/james/Documents/GitHub/PHYS451-MPhys_project/q22_diff/q22_diff_1000.pkl"
try:    
    infile = open(pkl_path,'rb')       
    q22_diff = pickle.load(infile)
    infile.close()
    print("Prior Loaded - "+pkl_path)
    
except FileNotFoundError:
    print(pkl_path+" not found")
    exit()
#print(max(q22_diff))
q22_diff=[np.log10(i) for i in q22_diff if i!=0]    
hist=plt.figure()
plt.xlabel(r"$log_{10}(q_{22_{hist}}/q_{22_{param}})$")
plt.hist(q22_diff, bins=200)
hist.show()