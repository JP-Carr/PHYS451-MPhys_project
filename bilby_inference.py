import os
import subprocess as sp
from collections import OrderedDict

import corner
import h5py
import matplotlib.font_manager as font_manager
import numpy as np
from astropy.utils.data import download_file
from bilby.core.prior import Uniform
from comparitors import comparisons
from cwinpy import HeterodynedData
from cwinpy.pe import pe
from lalinference import LALInferenceHDF5PosteriorSamplesDatasetName
from lalinference.io import read_samples
from matplotlib.lines import Line2D
import pickle

def pickler(path,obj):
    """
    Pickles an object and saves it to a chosen location

    Parameters
    ----------
    path : str
        Path of directory to save .pck file.
    obj : Python object 
        Object to pickle.

    Returns
    -------
    None.

    """
    outfile = open(path,'wb')
    pickle.dump(obj,outfile)
    outfile.close()
    print(path+" pickled")



parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       1.1e-25
COSIOTA  0.01
PSI      1.1
PHI0     2.4
"""

detector = "H1"  # the detector to use
asd = 1e-24  # noise amplitude spectral density


label = "single_detector_software_injection_linear"
outdir=""

parfile = os.path.join(outdir, "{}.par".format(label))
with open(parfile, "w") as fp:
    fp.write(parcontent)

times = np.linspace(1000000000.0, 1000086340.0, 1440)

het = HeterodynedData(
    times=times,
    par=parfile,
    injpar=parfile,
    inject=True,
    fakeasd=asd,
    detector=detector,
)

# output the data
hetfile = os.path.join(outdir, "{}_data.txt".format(label))
het.write(hetfile)


phi0range = [0.0, np.pi]
psirange = [0.0, np.pi / 2.0]
cosiotarange = [-1.0, 1.0]
h0range = [0.0, 1e-23]

# set prior for lalapps_pulsar_parameter_estimation_nested
priorfile = os.path.join(outdir, "{}_prior.txt".format(label))
priorcontent = """H0 uniform {} {}
PHI0 uniform {} {}
PSI uniform {} {}
COSIOTA uniform {} {}
"""
with open(priorfile, "w") as fp:
    fp.write(priorcontent.format(*(h0range + phi0range + psirange + cosiotarange)))

# set prior for bilby
priors = OrderedDict()
priors["h0"] = Uniform(h0range[0], h0range[1], "h0", latex_label=r"$h_0$")
priors["phi0"] = Uniform(
    phi0range[0], phi0range[1], "phi0", latex_label=r"$\phi_0$", unit="rad"
)
priors["psi"] = Uniform(
    psirange[0], psirange[1], "psi", latex_label=r"$\psi$", unit="rad"
)
priors["cosiota"] = Uniform(
    cosiotarange[0], cosiotarange[1], "cosiota", latex_label=r"$\cos{\iota}$"
)


Nlive = 1024  # number of nested sampling live points

runner = pe(
    data_file=hetfile,
    par_file=parfile,
    prior=priors,
    detector=detector,
    sampler="nestle",
    sampler_kwargs={"Nlive": Nlive, "walks": 40},
    outdir=outdir,
    label=label,
)

result = runner.result

pickler("bilby_result.pkl", result)