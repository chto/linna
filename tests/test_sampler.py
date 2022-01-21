import pytest
from linna.sampler import *
import numpy as np
def test_zeus():
    def log_prob(x):
       return - 0.5 * np.sum(ivar * x**2.0)
    nsteps, nwalkers, ndim = 1000, 10, 2
    ivar = 1.0 / np.random.rand(ndim)
    start = np.random.randn(nwalkers,ndim)
    samp = ZeusSampler(log_prob, ndim, nwalkers,  x0=start)     
    outdir = "tests/out/mcmcout_zeus/"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    samp.sample(None, nsteps, outdir=outdir, progress=False, overwrite=False, ntimes=10, tautol=0.01, incremental=True)


if __name__=="__main__":
    test_zeus()
