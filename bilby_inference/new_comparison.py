import os
import bilby
import cwinpy
import h5py
import numpy as np
from bilby.core.result import read_in_result
from lalinference import LALInferenceHDF5PosteriorSamplesDatasetName
from lalinference.io import read_samples
from scipy.spatial.distance import jensenshannon
from scipy.stats import combine_pvalues, ks_2samp
import torch
from HeterodynedData import generate_het

parameter_conversion=["H0", "PHI0", "PSI", "COSIOTA"]

#observation_parameters={"H0*1e25": 5.12e-23 *1e25,   # paramters for test GW
 #                       "phi0": 2.8,
  #                      "cosiota": 0.3,
   #                     "psi": 0.82
    #                    }


def credible_interval(samples, ci=0.90):
    # get the given percentage credible interval about the median
    return np.quantile(samples, [0.5 - 0.5 * ci, 0.5 + 0.5 * ci])

def comparisons(label, outdir, grid, priors, NN_posterior, injection_parameters, cred=0.9):
    """
    Perform comparisons of the evidence, parameter values, confidence
    intervals, and Kolmogorov-Smirnov test between samples produced with
    lalapps_pulsar_parameter_estimation_nested and cwinpy.
    """

     
    result = read_in_result(outdir=outdir, label=label)


    for p in priors.keys():
        samples=len(result.posterior[p])
        break

  #  samples=len(result.posterior["h0"][:,1])
    print(samples)
    observation=torch.from_numpy(generate_het(H0=injection_parameters["h0"], PHI0=injection_parameters["phi0"], PSI=injection_parameters["psi"], COSIOTA=injection_parameters["cosiota"]).data)
    post = NN_posterior.sample((samples,), x=observation)
    

    pvalues = []
    jsvalues = []
    for p in priors.keys():

        psample=post[:, parameter_conversion.index(p.upper())].numpy()
        print(len(psample), len(result.posterior[p]))
        _, pvalue = ks_2samp(psample, result.posterior[p])
        pvalues.append(pvalue)

        # calculate J-S divergence
        bins = np.linspace(
            np.min([np.min(psample), np.min(result.posterior[p])]),
            np.max([np.max(psample), np.max(result.posterior[p])]),
            100,
        )

        hp, _ = np.histogram(psample, bins=bins, density=True)
        hq, _ = np.histogram(result.posterior[p], bins=bins, density=True)
        jsvalues.append(jensenshannon(hp, hq) ** 2)

    return jsvalues