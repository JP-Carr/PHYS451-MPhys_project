from cwinpy.pe.simulation import PEPulsarSimulationDAG
from bilby.core.prior import PriorDict, Uniform, Exponential, Sine
import numpy as np

# set the Q22 distribution
mean = 1e33
ampdist = Exponential(name="q22", mu=mean)

# set the prior distribution for use in parameter estimation
prior = PriorDict({
    "q22": Uniform(0.0, 1e40, name="q22"),
    "iota": Sine(name="iota"),
    "phi0": Uniform(0.0, np.pi, name="phi0"),
    "psi": Uniform(0.0, np.pi / 2, name="psi"),
})

# set the detectors to use (and generate fake data for)
detectors = ["H1", "L1"]

# generate the population
run = PEPulsarSimulationDAG(ampdist=ampdist, prior=prior, npulsars=1000, detector=detectors)
