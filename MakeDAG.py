from cwinpy.pe.simulation import PEPulsarSimulationDAG
from bilby.core.prior import PriorDict, Uniform, Exponential, Sine
import numpy as np
import datetime
from os import mkdir, rmdir

test_mode=True
num_pulsars=500

current_time=datetime.datetime.now()
if test_mode==True:
    output_dir="DAGout/test{}".format(num_pulsars)
   # rmdir(output_dir)
    mkdir(output_dir)
else:
    output_dir="DAGout/{}{}_{}{}".format(current_time.day, current_time.month, current_time.hour, current_time.minute)
print(output_dir)
# set the Q22 distribution
mean = 1e33
ampdist = Exponential(name="q22", mu=mean)

# set the prior distribution for use in parameter estimation
prior = PriorDict({
    "h0": Uniform(0.0, 1e-22, name="h0"),
    "iota": Sine(name="iota"),
    "phi0": Uniform(0.0, np.pi, name="phi0"),
    "psi": Uniform(0.0, np.pi / 2, name="psi"),
})

# set the detectors to use (and generate fake data for)
detectors = ["H1", "L1"]

# generate the population
run = PEPulsarSimulationDAG(ampdist=ampdist, prior=prior, npulsars=num_pulsars, detector=detectors, basedir=output_dir)
print("created {} pulsars".format(num_pulsars))
