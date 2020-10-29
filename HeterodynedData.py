import numpy as np
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, TargetedPulsarLikelihood
from matplotlib import pyplot as plt
import sys
import pickle as pkl

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
#times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
times = np.linspace(1000000000.0, 1000086340.0, 1440)
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
het.write("data.txt")

prior = {}
prior["h0"] = Uniform(0.0, 1e-22, "h0")
prior["phi0"] = Uniform(0.0, np.pi, "phi0")
priordic = PriorDict(prior)
print(type(priordic))
pkl.dump("")