import pytest
import numpy as np
from linna.main import ml_sampler,ml_sampler_core
from linna.util import *
from linna.nn import *
ndim = 2 
init =  np.random.uniform(size=ndim)
#covariance matrix 
cov = np.diag([0.5, 0.2])
means = np.array([0.1, 1])

priors = []
for i in range(ndim):
    priors.append({
        'param': 'test_{0}'.format(i),
        'dist': 'flat',
        'arg1': -2.,
        'arg2': 2.
    })

def theory(x, outdirs):
    x_new = deepcopy(x[1])
    return x_new
nwalkers = 4 #Number of mcmc walker
pool = None
ntrainArr = [1]
nvalArr = [1]
nkeepArr = [2]
ntimesArr = [1]
ntautolArr = [0.1]
temperatureArr =  [2.0]
params = {}
params["trainingoption"] = 1
params["num_epochs"] = 10
params["batch_size"] = 5
ypositive=False
dolog10index = None
outdir = os.path.abspath(os.getcwd())+"/out/2dgaussian_Fulltconn/"
gpunode = None
def testmain():
    chain, logprob = ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, outdir, theory, priors, means, cov,  init, pool, nwalkers, "cuda", dolog10index, ypositive, temperatureArr, omegab2cut=None, docuda=False, tsize=1, gpunode=None, nnmodel_in=ChtoModelv2, params=params, method="emcee")


if __name__ == '__main__':
    testmain()
