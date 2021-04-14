import os
from collections import OrderedDict
import torch
import corner
import numpy as np
from bilby.core.prior import Uniform
from cwinpy import HeterodynedData
from cwinpy.pe import pe
import pickle
import time
import datetime
from HeterodynedData import generate_het
from zplib.scalar_stats.compare_distributions import  js_metric
from copy import deepcopy
from SBI_inference import observe
from new_comparison import comparisons
from sys import exit


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

start=time.time()

SNR=5

injection_parameters = OrderedDict()
injection_parameters["h0"] = 5.12e-23
injection_parameters["phi0"] = 0.777
injection_parameters["psi"] = 0.515
injection_parameters["cosiota"] = 0.820


detector = "H1"  # the detector to use
asd = injection_parameters["h0"]/5 #1e-24  # noise amplitude spectral density

parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       {}
COSIOTA  {}
PSI      {}
PHI0     {}
""".format(injection_parameters["h0"], injection_parameters["cosiota"], injection_parameters["psi"] ,injection_parameters["phi0"])
#print(parcontent)

label = "single_detector_software_injection_linear"
current_time=datetime.datetime.now()
outdir="output/{}{}_{}{}".format(current_time.day, current_time.month, current_time.hour, current_time.minute)
print(outdir)
os.mkdir(outdir)

parfile = os.path.join(outdir, "{}.par".format(label))
with open(parfile, "w") as fp:
    fp.write(parcontent)

times = np.linspace(1000000000.0, 1000086340.0, 1440)
start_cwinpy=time.time()
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

# run lalapps_pulsar_parameter_estimation_nested

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
    
  
gridpoints = 35
grid_size = dict()
for p in priors.keys():
    grid_size[p] = np.linspace(
        np.min(result.posterior[p]), np.max(result.posterior[p]), gridpoints
    )



grunner = pe(
      data_file=hetfile,
      par_file=parfile,
      prior=priors,
      detector=detector,
      outdir=outdir,
      label=label,
      grid=True,
      grid_kwargs={"grid_size": grid_size},
  )
  
grid = grunner.grid
   
    
print(time.time()-start_cwinpy)
    
fig = result.plot_corner(save=False, parameters=injection_parameters, color="b")
#exit()
"""
fig = corner.corner(
    postsamples,
    fig=fig,
    color="r",
    bins=50,
    smooth=0.9,
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    fill_contours=True,
    hist_kwargs={"density": True},
)
"""
start_sbi=time.time()
posterior_path="/home/james/Documents/GitHub/PHYS451-MPhys_project/posteriors/posterior80000_SNPEnew.pkl"
infile = open(posterior_path,'rb')       #Try to load relevent posterior 
posterior = pickle.load(infile)
infile.close()
print("Prior Loaded - "+posterior_path)
"""
ob_het=generate_het(H0=injection_parameters["h0"]*1e25, PHI0=injection_parameters["phi0"],PSI=injection_parameters["psi"] , COSIOTA=injection_parameters["cosiota"],  fakeasd=asd).data
observation=torch.from_numpy(np.concatenate((ob_het.real,ob_het.imag)))
samples = posterior.sample((10000,), x=observation)
"""
samples=observe(posterior, h0=injection_parameters["h0"]*1e25, phi0=injection_parameters["phi0"], psi=injection_parameters["psi"], cosiota=injection_parameters["cosiota"], plot=False, num_samples=10000)
print(time.time()-start_sbi)
samples[:,0]=samples[:,0]/1e25


store=deepcopy(samples[:,-2])
samples[:,-2]=samples[:,-1]
samples[:,-1]=store
print("---------------------------")
print(samples)
#exit()
axes = fig.get_axes()
axidx = 0
count=0
for p in priors.keys():
    print(p)
    y=np.exp(grid.marginalize_ln_posterior(not_parameters=p) - grid.log_evidence)
 #   print("///////////////////////////")
    axes[axidx].plot(
        grid.sample_points[p],
        np.exp(grid.marginalize_ln_posterior(not_parameters=p) - grid.log_evidence),
        "k--",
    )
    #axes[axidx].plot(samples[:,count],y)
    print(grid.sample_points[p])
    print(samples[:,count])
    print("JS: "+str(js_metric(grid.sample_points[p], samples[:,count])))
    print("//////////////////")
    count+=1
    axidx += 5
"""
_ = corner.corner(
    samples[:,0:4],
    fig=fig,
    color="g",
    bins=50,
    smooth=0.9,
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    fill_contours=True,
    hist_kwargs={"density": True},
    )   
"""
fig.savefig(os.path.join(outdir, "{}_corner.png".format(label)), dpi=150)
print("\nRuntime = {}s".format(round(time.time()-start,2)))






print(comparisons(label, outdir, grid, priors, posterior, injection_parameters, cred=0.9))