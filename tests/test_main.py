import pytest
import numpy as np
from linna.main import ml_sampler,ml_sampler_core
from linna.util import *
from linna.nn import *
import numpy.testing as npt
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
ntrainArr = [20]
nvalArr = [5]
nkeepArr = [1]
ntimesArr = [2]
ntautolArr = [0.5]
temperatureArr =  [1.0]
meanshiftArr=[100]
stdshiftArr = [100]
params = {}
params["trainingoption"] = 1
params["num_epochs"] = 10
params["batch_size"] = 5
ypositive=False
dolog10index = None
gpunode = None

def testmain():
    outdir = os.path.abspath(os.getcwd())+"/out/2dgaussian_Fulltconn/"
    chain, logprob = ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, meanshiftArr, stdshiftArr, outdir, theory, priors, means, cov,  init, pool, nwalkers, "cuda", dolog10index, ypositive, temperatureArr, omegab2cut=None, docuda=False, tsize=1, gpunode=None, nnmodel_in=ChtoModelv2, params=params, method="emcee")

def test_reading():
    outdir = os.path.abspath(os.getcwd())+"/test_data/2dgaussian_Fulltconn/"
    chain, logprob = ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, meanshiftArr, stdshiftArr, outdir, theory, priors, means, cov,  init, pool, nwalkers, "cuda", dolog10index, ypositive, temperatureArr, omegab2cut=None, docuda=False, tsize=1, gpunode=None, nnmodel_in=ChtoModelv2, params=params, method="emcee")
    npt.assert_almost_equal(np.mean(chain),  0.15151080063411168, decimal=5)
    npt.assert_almost_equal(np.std(chain), 0.9633211647095377, decimal=5)




if __name__ == '__main__':
    testmain()
    test_reading()
