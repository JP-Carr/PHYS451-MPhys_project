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

     
 


    observation=torch.from_numpy(generate_het(H0=injection_parameters["h0"], PHI0=injection_parameters["phi0"], PSI=injection_parameters["psi"], COSIOTA=injection_parameters["cosiota"]).data)
    post = NN_posterior.sample((10000,), x=observation)
    
    """    
    lppenfile = os.path.join(outdir, "{}_post.hdf".format(label))

    # get posterior samples
    post = read_samples(
        lppenfile, tablename=LALInferenceHDF5PosteriorSamplesDatasetName
    )

    # get uncertainty on ln(evidence)
    info = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "information_nats"
    ]
    nlive = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "number_live_points"
    ]
    evsig = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_evidence"
    ]
    evnoise = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_noise_evidence"
    ]
    """
    
   # everr = np.sqrt(info / nlive)  # the uncertainty on the evidence

    # read in cwinpy results
    result = read_in_result(outdir=outdir, label=label)

    # comparison file
  #  comparefile = os.path.join(outdir, "{}_compare.txt".format(label))

    # get grid-based evidence
   # if grid is not None:
    #    grid_evidence = grid.log_evidence



    # calculate the Kolmogorov-Smirnov test for each 1d marginalised distribution,
    # and the Jensen-Shannon divergence, from the two codes. Output the
    # combined p-value of the KS test statistic over all parameters, and the
    # maximum Jensen-Shannon divergence over all parameters.
  #  values[idx] = np.inf
    pvalues = []
    jsvalues = []
    for p in priors.keys():
        
        #print(p)
        print(p.upper())
     #   print(post[p.upper()])
        
        psample=post[:, parameter_conversion.index(p.upper())].numpy()
       # print(psample)
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