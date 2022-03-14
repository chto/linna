=====
LINNA
=====


.. image:: https://img.shields.io/pypi/v/linna.svg
        :target: https://pypi.python.org/pypi/linna

.. image:: https://github.com/chto/linna/actions/workflows/check.yml/badge.svg
        :target: https://github.com/chto/linna/actions/workflows/check.yml

.. image:: https://readthedocs.org/projects/linna/badge/?version=latest
        :target: https://linna.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status
        
.. image:: https://img.shields.io/badge/arXiv-2003.05583-blue.svg
        :target: https://arxiv.org/abs/2203.05583

**Linna (Likelihood Inference Neural Network Accelerator) is a tool to accelerate Bayesian posterior inferences using artificial neural networks.**

- Linna automatically builds training data, trains the neural network, and produces a Markov chain that samples the posterior.
- Linna reduces the runtime of survey cosmological analyses of the Dark Energy Survey by a factor of 8-50.
- Linna is verified to enable accurate and efficient sampling for Vera Rubin Observatory's Legacy Survey of Space and Time (LSST) year ten multi-probe analyses.
- Linna is explicitly verified for the following three multi-probe analyses:
    - 3x2pt, a joint analysis of galaxy clustering, galaxy-galaxy lensing, and cosmic shear.
    - 4x2pt+N, a joint analysis of cluster--galaxy cross correlations, cluster lensing, cluster clustering, and cluster abundances.
    - 6x2pt+N, a joint analysis of data vectors in 3x2pt and 4x2pt+N.



Documentation
-------------
Read the docs at https://linna.readthedocs.io/en/latest/readme.html#documentation

Installation
-------------

::

    git clone https://github.com/chto/linna.git
    cd linna 
    python setup.py install

Attribution
-----------
Please cite the paper below if you find LINNA useful:
::

    @article{linna2022,
    author = {Chun-Hao To and Eduardo Rozo and Elisabeth Krause and Hao-Yi Wu and Risa H. Wechsler and Andrés N. Salcedo},
    title = {LINNA: Likelihood Inference Neural Network Accelerator},
    year = {2022},
    journal={arXiv preprint arXiv:2203.05583}
    }



Example
-------
For example, if you want to sample a 33 dimensional gaussian spaces, you can do 

.. code-block:: python
 
    import numpy as np
    import matplotlib.pyplot as plt 
    from linna.main import ml_sampler
    from linna.util import *
    #Define gaussian 
    ndim = 33
    init =  np.random.uniform(size=ndim)
    means = np.random.uniform(size=ndim)
    cov = np.diag(0.1*np.random.uniform(size=ndim))
    priors = []
    for i in range(ndim):
        priors.append({
            'param': 'test_{0}'.format(i),
            'dist': 'flat',
            'arg1': -5.,
            'arg2': 5.
        })
    def theory(x, outdirs):
        x_new = deepcopy(x[1])
        return x_new
    #LINNA
    nwalkers = 4 #Number of mcmc walker
    pool = None
    outdir = os.path.abspath(os.getcwd())+"/out/2dgaussian/"
    chain, logprob = ml_sampler(outdir, theory, priors, means, cov, init, pool, nwalkers, gpunode=None, nepoch=101)
    


