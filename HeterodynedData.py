import numpy as np
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, TargetedPulsarLikelihood
from matplotlib import pyplot as plt
import sys
import pickle as pkl
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

run=False

def generate_het(H0=1.1e-25*1e25,COSIOTA=0.01,PSI=1.1,PHI0=2.4):
  #  print(PSI)
   # print("----------")
    #print(PHI0)
    use_parfile=False
    
    parname="J0123+3456"
    
    
    if use_parfile==True:
    
        parcontent = """\
        PSRJ     {}
        RAJ      01:23:45.6789
        DECJ     34:56:54.321
        F0       567.89
        F1       -1.2e-12
        PEPOCH   56789
        H0       {}
        COSIOTA  {}
        PSI      {}
        PHI0     {}
        """.format(parname,H0,COSIOTA,PSI,PHI0)
        
        par = parname+".par"
        with open(par, "w") as fp:
            fp.write(parcontent)
    else:
    
      #  print(0)
        par = PulsarParametersPy()  # create an empty object
        par["PSRJ"] = parname  # give it a name
        par["RAJ"] = 1.8  # give it a right ascension (in rads)
        par["DECJ"] = 0.54  # give it a declination (in rads)
        par["F"] = [567.89] # give it a frequency (in Hz)
        par["H0"] = H0  # give it a gravitational-wave amplitude (between 0 and 1e-22)
        par["COSIOTA"] = COSIOTA  # give it a value of cos(iota) (between -1 and 1)
        par["PSI"] = PSI  # give it a value of the polarisation angle (in rads) (between 0 and pi)
        par["PHI0"] = PHI0  # give it a value of the initial phase (in rads) (between 0 and pi/2)
     #   parcontent["PEPOCH"] = 1000000000
       # print(1)
       # print(parcontent)
    
    detector = "H1"  
   # print(2)
    times = np.linspace(1000000000.0, 1000086340.0, 1440)
    times=np.concatenate((times,times))
    het = HeterodynedData(
        times=times,
        inject=True,
        par=par,
        injpar=par,
        fakeasd=1e-24,
        detector=detector,
        bbminlength=1e10000
    )
  #  print(3)
    return het



if __name__=="__main__" and 1==2:
    x=generate_het()
    print(x.data)
    times = np.linspace(1000000000.0, 1000086340.0, 1440)
    plt.figure()
    plt.plot(times,x.data)
    plt.show()
    
    

if __name__=="__main__" and run==True:
    
    parcontent = """\
    PSRJ     J0123+3456
    RAJ      01:23:45.6789
    DECJ     34:56:54.321
    F0       567.89
    F1       -1.2e-12
    PEPOCH   56789
    H0       5.12e-25
    COSIOTA  0.3
    PSI      1.1
    PHI0     2.4
    """
    
    # add content to the par file
    parfile = "J0123+3456.par"
    with open(parfile, "w") as fp:
        fp.write(parcontent)
        
    # create some fake heterodyned data
    detector = "H1"  # the detector to use
    times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
    times=np.concatenate((times,times))
    het = HeterodynedData(
        times=times,
        inject=True,
        par=parfile,
        injpar=parfile,
        fakeasd=1e-24,
        detector=detector,
    )
    
    #print(type(het.par))
    #np.save("het.npy",(het.data))
 #   het.write("data.txt")
    """
    prior = {}
    prior["h0"] = Uniform(0.0, 1e-22, "h0")
    prior["phi0"] = Uniform(0.0, np.pi, "phi0")
    priordic = PriorDict(prior)
    print(type(priordic))
   # pkl.dump("")
    """

if __name__=="__main__":
    print("--------------------")
    x=generate_het(PHI0=2.4, PSI=1.1).data
    y=generate_het(PSI=0.82, PHI0=0.74).data
    #y=generate_het(PSI=0.66, PHI0=0.5).data
  #  print(np.mean(x-y))
   # print(x.real)
    z=np.concatenate((x.real, x.imag ))
    import torch
    print(torch.from_numpy(z))