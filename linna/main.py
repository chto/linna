import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
import glob
import pickle
import sys
import emcee
from linna.nn import *
from scipy.special import erf
from scipy.stats import chi2
import io
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import mkldnn as mkldnn_utils
from linna.util import *
import tempfile


def ml_sampler(outdir, theory, priors, data, cov, init, pool, nwalkers, gpunode, omegab2cut=None, nepoch=4500, method="zeus", nbest=None, chisqcut=None, loglikelihoodfunc=None):
    """
    LINNA main function with hyperparameters set to values described in To et al. 2022

    Args:
        outdir (string): output directory 
        theory (function): theory model 
        priors (dict of str: [float, float]): string can be either flat or gauss. If the string is 'flat', [a,b] indicates the lower and upper limits of the prior. If the string is 'gauss', [a,b] indicates the mean and sigma. 
        data (1d array): float array, data vector 
        cov (2d array): float array, covariance matrix
        init (ndarray): initial guess of mcmc,
        pool (mpi pool, optional): a mpi pool instance that can do pool.map(function, iterables). 
        nwalkers (int) number of mcmc walkers
        gpunode (string): name of gpu node
        omegab2cut (list of int): 2 elements containing the lower and upper limits of omegab*h^2
        nepoch (int, optional): maximum number of epoch for the neural network training
        method (string, optional): Samplers. LINNA supports `emcee` and `zeus`(default)
        nbest (int or list of int): number of points to include in the training set per iteration according to the optimizer
        chisqcut (float, optional): cut the training data if there chisq is greater than this value 
        loglikelihoodfunc (callable, optional): function of model, data , inverse of covariance matrix and return the log liklihood value. If None, then use gaussian likelihood 
    Returns:
        nd array: MCMC chain 
        1d array: log probability of MCMC chain
        
    """
    ntrainArr = [10000, 10000, 10000, 10000]
    nvalArr = [500, 500, 500, 500]
    if method=="emcee":
        nkeepArr = [2, 2, 5, 4]
        ntimesArr = [5, 5, 10, 15]
        ntautolArr = [0.03, 0.03, 0.02, 0.01]
        temperatureArr =  [4.0, 2.0, 1.0, 1.0]
        meanshiftArr = [0.2, 0.2, 0.2, 0.2]
        stdshiftArr = [0.15,0.15,0.15,0.15]
    elif method=="zeus":
        nkeepArr = [2, 2, 5, 5]
        ntimesArr = [5, 5, 10, 50]
        ntautolArr = [0.03, 0.03, 0.02, 0.01]
        temperatureArr =  [4.0, 2.0, 1.0, 1.0]
        meanshiftArr = [0.2, 0.2, 0.2, 0.2]
        stdshiftArr = [0.15,0.15,0.15,0.15]
    else:
        raise NotImplementedError(method)
    dolog10index = None
    ypositive = False 
    device = "cuda"
    docuda=False
    tsize=1
    nnmodel_in = ChtoModelv2
    params = {}
    params["trainingoption"] = 1 
    params["num_epochs"] = nepoch
    params["batch_size"] = 500
    return ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, meanshiftArr, stdshiftArr, outdir, theory, priors, data, cov,  init, pool, nwalkers, device, dolog10index, ypositive, temperatureArr, omegab2cut, docuda, tsize, gpunode, nnmodel_in, params, method, nbest=nbest, chisqcut=chisqcut, loglikelihoodfunc=loglikelihoodfunc) 

def ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, meanshiftArr, stdshiftArr, outdir, theory, priors, data, cov,  init, pool, nwalkers, device, dolog10index, ypositive, temperatureArr, omegab2cut=None, docuda=False, tsize=1, gpunode=None, nnmodel_in=None, params=None, method="emcee", nbest=None, chisqcut=None, loglikelihoodfunc=None, nsigma=3):
    """
    LINNA main function 

    Args:
        ntrainArr (int array): number of training data per iteration 
        nvalArr (int array): number of validation data per iteration 
        nkeepArr (int array): number of autocorrelation time to be kept 
        ntimesArr (int array): number of autocorrelation time to stop mcmc
        ntautolArr (float array): error limit of autocorrelation time 
        meanshiftArr (float array): limit on mean shift of parameter estimation from the first and second half of the chain 
        stdshiftArr (float array): limit on std shift of parameter estimation from the first and second half of the chain
        outdir (string): output directory 
        theory (function): theory model 
        priors (dict of str: [float, float]): string can be either flat or gauss. If the string is 'flat', [a,b] indicates the lower and upper limits of the prior. If the string is 'gauss', [a,b] indicates the mean and sigma. 
        data (1d array): float array, data vector 
        cov (2d array): float array, covariance matrix
        init (ndarray): initial guess of mcmc,
        pool (object): mpi4py pool instance 
        nwalkers (int) number of mcmc walkers
        device (string): cpu or gpu
        dolog10index (int array): index of parameters to do log10 
        ypositive (bool): whether the data vector is expected to be all positive 
        temperatureArr (float array): temperature parameters for each iteration
        omegab2cut (list of int): 2 elements containing the lower and upper limits of omegab*h^2  
        docuda (bool): whether do gpu for evaluation 
        tsize (int, optional): number of cores for training 
        gpunode (string): name of gpu node
        nnmodel_in (string): instance of neural network model 
        params (dictionary): dictionary of parameters 
        method (string): sampling method 
        nbest (int or list of int): number of points to include in the training set per iteration according to the optimizer
        chisqcut (float, optional): cut the training data if there chisq is greater than this value 
        loglikelihoodfunc (callable): function of model, data , inverse of covariance matrix and return the log liklihood value 
        nsigma (float): the training point in the first iteration will be generated within nsigma of the gaussian prior 
    Returns:
        nd array: MCMC chain 
        1d array: log probability of MCMC chain
        
    """
    ndim = len(init)
    sigma = np.sqrt(np.diag(cov))
    inv_cov = np.linalg.inv(cov)    
    prior_range = []
    for item in priors:
        if item['dist'] == 'flat':
            prior_range.append([item['arg1'], item['arg2']])
        elif item['dist'] == 'gauss':
            prior_range.append([item['arg1']-5*item['arg2'], item['arg1']+5*item['arg2']])
        else:
            print("not implement dist : {0}".format(item['dist']), flush=True)
            assert(0)
    transform = Transform(priors)
    invtransform = invTransform(priors)
    init = invtransform(init) 
    if method=="emcee":
        filename = "chemcee_256.h5"
    elif method == "zeus":
        filename = "zeus_256.h5"
    else:
        raise NotImplementedError(method)
    for i, (nt, nv, nk, ntimes, tautol, temperature, meanshift, stdshift) in enumerate(zip(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, temperatureArr, meanshiftArr, stdshiftArr)):
        if isinstance(nbest, list):
            nbest_in = nbest[i]
            if nbest_in <=0:
                nbest_in = None
        else:
            nbest_in = nbest
        if nbest_in is not None:
            tempdir = tempfile.TemporaryDirectory()
            def negloglike(x):
                d = data-theory([-1,x], tempdir)
                return d.dot(inv_cov.dot(d))
        else:
            negloglike=None
        temperature = temperature**2
        print("#"*100)
        print("iteration: {0}".format(i), flush=True)
        print("#"*100)
        outdir_in = os.path.join(outdir, "iter_{0}/".format(i))
        if i==0:
            chain=None
        else:
            chain_name = os.path.join(os.path.join(outdir, "iter_{0}/".format(i-1)), filename[:-3])
            if os.path.isfile(chain_name+".h5"):
                chain_name = chain_name+".h5"
                chain, _temp, _temp2= read_chain_and_cut(chain_name.format(i-1), nk, ntimes, method=method)
            else:
                chain_name = chain_name+".txt"
                chain = np.loadtxt(chain_name)[-100000:,:-1]
        #Generate training 
        ntrain = nt
        nval = nv
        nnsampler = NN_samplerv1(outdir_in, prior_range) 
        if "trainingoption" in params:
            options = params['trainingoption']
        else:
            options = 0
        generate_training_point(theory, nnsampler, pool, outdir_in, ntrain, nval, data, inv_cov, chain, nsigma=nsigma, omegab2cut=omegab2cut, options=options,  negloglike= negloglike, nbest_in=nbest_in, chisqcut=chisqcut)
        chain = None
        del chain 
        if i!=0:
            try:
                del _temp
                del _temp2
            except:
                pass
        gc.collect()
        if (pool is None) or pool.is_master():
            outdir_list = [os.path.join(outdir, "iter_{0}/".format(m)) for m in range(int(i+1))]

            f = open(outdir_list[-1]+"/model_pickle.pkl", 'wb')
            pickle.dump(train_NN, f) 
            f.close()
            f = open(outdir_list[-1]+"/model_args.pkl", 'wb')
            if gpunode is not None:
                docuda=True
            else:
                docuda=torch.cuda.is_available()
            pickle.dump([nnsampler, cov, inv_cov, sigma, outdir_in, outdir_list, data, dolog10index, ypositive, False, 2, temperature, docuda, None, 1, nnmodel_in, params, nbest_in is not None], f) 
            f.close()
            if not os.path.isfile(outdir_list[-1] + "/finish.pkl"): 
                if gpunode == 'automaticgpu':
                    while(True):
                        gpufile = os.path.join(outdir, "gpunodeinfo.pkl")
                        try:
                            if os.path.isfile(gpufile):
                                with open(gpufile, 'rb') as f:
                                    gpuinfo = pickle.load(f)
                                gpunode = gpuinfo["nodename"]
                                break 
                        except:
                            pass
                if gpunode is not None:
                    print("running gpu on {0}".format(gpunode), flush=True)
                    os.system("cat {2}/train_gpu.py | ssh {0} python - {1} {3}".format(gpunode, outdir_list[-1], os.path.dirname(os.path.abspath(__file__)), "cuda"))
                    while(1):
                        if  os.path.isfile(outdir_list[-1] + "/finish.pkl"):
                            break
                else:
                    os.system("python {1}/train_gpu.py {0} {2}".format(outdir_list[-1], os.path.dirname(os.path.abspath(__file__)), "nocuda"))
                    while(1):
                        if  os.path.isfile(outdir_list[-1] + "/finish.pkl"):
                            break

        #Retrieve model  
        model, y_invtransform_data = retrieve_model(outdir_in, len(init), len(data), nnmodel_in)
        if not docuda:
            model.model = model.model.to(memory_format=torch.channels_last)
            model.MKLDNN=True
        #Do MCMC
        if os.path.isfile(os.path.join(outdir_in, filename)):
            continue
        invcov_new = torch.from_numpy(inv_cov.astype(np.float32)).to('cpu').detach().clone().requires_grad_()
        data_new = torch.from_numpy(data.astype(np.float32)).to('cpu').detach().clone().requires_grad_()
        if loglikelihoodfunc is None:
            loglikelihoodfunc = gaussianlogliklihood
        log_prob = Log_prob(data_new, invcov_new, model, y_invtransform_data, transform, temperature, nograd=True, loglikelihoodfunc=loglikelihoodfunc) 
        dlnp = None
        ddlnp = None
        if pool is not None:
            pool.noduplicate=True
        run_mcmc(nnsampler, outdir_in, method, ndim, nwalkers, init, log_prob, dlnp=dlnp, ddlnp=ddlnp, pool=pool, transform=transform, ntimes=ntimes, tautol=tautol, meanshift=meanshift, stdshift=stdshift, nk=nk)
        if pool is not None:
            pool.noduplicate_close() 
    chain_name = os.path.join(os.path.join(outdir, "iter_{0}/".format(len(ntrainArr)-1)), filename[:-3])
    if os.path.isfile(chain_name+".h5"):
        chain_name = chain_name+".h5"
        chain, log_prob_samples_x, reader = read_chain_and_cut(chain_name.format(len(ntrainArr)-1), nk, ntimes, method=method)
        log_prob_samples_x = reader.get_log_prob(discard=0, flat=True, thin=1)
    else:
        chain_name = chain_name+".txt"
        chain = np.loadtxt(chain_name)[-100000:,:-1]
        log_prob_samples_x = np.loadtxt(chain_name)[-100000:,-1]

    #Optional importance sampling 
    if 'nimp' in params.keys():
        if not os.path.isfile(outdir+"/samples_im.npy"):
            chain_name = os.path.join(os.path.join(outdir, "iter_{0}/".format(len(ntrainArr)-1)), filename[:-3])
            if os.path.isfile(chain_name+".h5"):
                chain_name = chain_name+".h5"
                chain, log_prob_samples_x, reader = read_chain_and_cut(chain_name.format(len(ntrainArr)-1), nk, ntimes, method=method, flat=True)
            else:
                chain_name = chain_name+".txt"
                chain = np.loadtxt(chain_name)[-100000:,:-1]
                log_prob_samples_x = np.loadtxt(chain_name)[-100000:,-1]
            select = np.random.randint(0, len(chain), params['nimp'])
            chain = chain[select]
            log_prob_samples_x = log_prob_samples_x[select]
            np.save(outdir+"/samples_im.npy", chain)
            np.save(outdir+"/log_prob_samples_x.npy", log_prob_samples_x)
        else:
            chain = np.load(outdir+"/samples_im.npy")
            log_prob_samples_x = np.load(outdir+"/log_prob_samples_x.npy")
        outimp = os.path.join(outdir, 'imp/')
        nnsampler = NN_samplerv1(outimp, prior_range)
        if not os.path.isdir(outimp):
            os.makedirs(outimp)
        if not os.path.isfile(outdir+"/theory.npy"):
            theory = nnsampler.generate_training_data(zip(range(len(chain)), chain), theory, pool=pool, args=[outimp])
            np.save(outdir+"/theory.npy", theory)
        else:
            theory = np.load(outdir+"/theory.npy")
        logprior=LogPrior(priors)
        log_prob_samples_x = log_prob_samples_x.flatten()
        logp = logp_theory_data(chain, theory, data, inv_cov, logprior)
        w = np.exp(logp-log_prob_samples_x)
        w[np.abs(np.log(w)-np.mean(np.log(w)))>2*np.std(np.log(w))]=0
        w = w/np.sum(w)
        np.save(outdir+"/weight_im.npy", [log_prob_samples_x.flatten(), logp, w])
    return chain, log_prob_samples_x





