import numpy as np
from copy import deepcopy
import os
import sys
path = "/home/users/chto/code/lighthouse/analysis/"
sys.path.append(path+"/../python/")
sys.path.append(path+"/../python/datavector/")
sys.path.append(path+"/../lib/")
import util_chto
import generate_data_vector
import  generate_batch_theory_datavector
from schwimmbad import MPIPool
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import numpy as np 
import sys
sys.path.append("/home/users/chto/code/lighthouse/python/datavector/")
import util_chto
sys.path.append("/home/users/chto/code/lighthouse/python/")
from pathlib import Path
import run_4x2ptN_wrapper
import fitsio
import cosmolike_libs_real_mpp_cluster 
import seaborn as sns
import pandas as pd
import argparse
import run_cosmolike_4x2ptN
import run_4x2ptN_wrapper
import pytest
import numpy.testing as npt
from tqdm import tqdm
import os
import random
import signal
import time
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from linna.main import ml_sampler,ml_sampler_core
from linna.util import *
from linna.nn import *
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



def get_prior_dic_init(param):
    (varied_params,
     cosmo_min, cosmo_fid, cosmo_max,
    nuisance_min, nuisance_fid, nuisance_max, cosmo_mean, cosmo_sigma, nuisance_mean, nuisance_sigma) = run_cosmolike_4x2ptN.parse_priors_and_ranges(param)
    nsigma=5
    prior_range = []
    init = []
    for ind, param in enumerate(varied_params):
        if param in cosmo_fid.names():
            sigma = getattr(cosmo_sigma,param)
            if sigma!=0:
                mean = getattr(cosmo_mean,param)
                prior_range.append({'param': param,
                                    'dist': 'gauss',
                                    'arg1': mean,
                                    'arg2': sigma})
            else:
                prior_range.append({'param': param,
                    'dist': 'flat',
                    'arg1': getattr(cosmo_min, param),
                    'arg2': getattr(cosmo_max, param)})
                mean = getattr(cosmo_fid, param)
            init.append(mean)
        elif param in nuisance_fid.names():
            ind_in = int(param.split("_")[-1])
            param_in="_".join(param.split("_")[:-1])
            sigma = getattr(nuisance_sigma,param_in)[ind_in]
            if sigma!=0:
                mean = getattr(nuisance_mean,param_in)[ind_in]
                prior_range.append({'param': param,
                                    'dist': 'gauss',
                                    'arg1': mean,
                                    'arg2': sigma})
            else:
                prior_range.append({'param': param,
                    'dist': 'flat',
                    'arg1': getattr(nuisance_min, param_in)[ind_in],
                    'arg2': getattr(nuisance_max, param_in)[ind_in]})
                mean = getattr(nuisance_fid, param_in)[ind_in]
            init.append(mean)
        else:
            print("no param : ", param)
            assert(0)
    return prior_range, np.array(init)


class Model_func:
    def __init__(self, datavector_writer, mask):
        self.datavector_writer = datavector_writer
        self.mask = mask
    def __call__(self, x, outdirs):
        data = x[1]
        data_file = os.path.join(outdirs, "data_{0}".format(x[0]))
        outfile = bytes(data_file, encoding='utf-8')
        if os.path.isfile(outfile):
            print("skip", outfile)
            return np.loadtxt(outfile.decode())[self.mask,1]
        print("work on: {0}".format(x[0]))
        sys.stdout.flush()
        self.datavector_writer(data, outfile)
        try: 
            data = np.loadtxt(outfile.decode())    
            if len(self.mask)>len(data):
                self.mask = self.mask[:len(data)]
            data = data[self.mask, 1]
        except:
            data = np.zeros_like(np.where(self.mask>0)[0])
        if len(data)==0:
            data = np.zeros_like(np.where(self.mask>0)[0])
        return data

def submitgpujob(allargs):
    outdir = allargs['outdir']
    qos = allargs['qos']
    timein = allargs['time'] 
    gpuconstraint = allargs['gpuconstraint']
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    joboutdir = os.path.join(outdir, "out/")
    if not os.path.isdir(joboutdir):
        os.makedirs(joboutdir)
    jobfile = os.path.join(outdir, "gpujob.sh")
    with open(jobfile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=gpu\n")
        fh.writelines("#SBATCH --output={0}/out.dat\n".format(joboutdir))
        fh.writelines("#SBATCH --error={0}/err.dat\n".format(joboutdir))
        fh.writelines("#SBATCH --time={0}\n".format(timein))
        #fh.writelines("#SBATCH --mem=4000\n")
        fh.writelines("#SBATCH -n 1\n")
        fh.writelines("#SBATCH --gres gpu:1\n")
        fh.writelines("#SBATCH  --constraint=\"{0}\"\n".format(gpuconstraint))
        fh.writelines("#SBATCH --gpu_cmode=shared\n")
        fh.writelines("#SBATCH -p {0}\n".format(qos))
        fh.writelines("srun python {0} {1} {2}\n".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpuscript.py"), outdir, timein))
    os.system("sbatch %s" %jobfile)
    

def main():
    import time
    start = time.time()
    method = sys.argv[1]
    params = util_chto.chto_yamlload(sys.argv[3], parent_dir=sys.argv[4])         

    outdir = params['outdir']
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
        except:
            pass
    gpunode  = sys.argv[2]
    device = "cuda"
    np.random.seed(42)
    nwalkers = 128
    if "nwalkers" in params:
        nwalkers = params['nwalkers']
    ntrainArr = params['ntrainArr']
    nvalArr =  params['nvalArr'] 
    nkeepArr =  params['nkeepArr']
    ntimesArr =  params['ntimesArr']
    ntautolArr =  params['ntautolArr']
    temperatureArr =   params['temperatureArr']
    nnmodel_in = eval(params['nnmodel'])

    if "mask_file" not in params:
        params['mask_file'] = os.path.join(outdir, "mask.dat")
        maskfile = params['mask_file']
        if 'DD_vector' in params.keys():
            try:
                DD_vector = np.loadtxt(os.path.join(params['base_dir'],params["DD_vector"]))[:,1]
                DD_cut = params['DD_cut']
            except:
                DD_vector=None
                DD_cut = None
        else:
            DD_vector = None
            DD_cut = None
        if rank==0:
            run_4x2ptN_wrapper.make_mask_4x2ptN(maskfile, params, notinitialized=True, DD_vector=DD_vector, DD_cut=DD_cut)
        else:
            while(1):
                try:
                    if len(np.loadtxt(maskfile))>0:
                        break
                except:
                    pass
    else:
        maskfile = params['mask_file']
    init_cosmolike = cosmolike_libs_real_mpp_cluster.Initlized_cosmolike(bytes(maskfile, encoding='utf-8'), params)
    init_cosmolike.set_cosmolike()
    priors, init = get_prior_dic_init(params)
    if "omegab2cut" in params:
        omegab2cut = params['omegab2cut']
    else:
        omegab2cut = [3,5,0.005,0.039]
    #try:
    pool = chtoPool(comm)
    #except:
    #    print("no MPI", flush=True)
    #    pool = None
    tsize=1
    mask = np.loadtxt(maskfile)[:,1]
    mask = mask>0
    if (pool is None) or pool.is_master():
            datavector_writer = generate_batch_theory_datavector.generate_base_file(params)
            theory = Model_func(datavector_writer, mask)
   
        
    def readcov(covin):
        cov  = np.zeros((int(np.max(covin[:,0]))+1, int(np.max(covin[:,0]))+1))
        for item in covin:
            cov[int(item[0]), int(item[1])] = item[-2]+item[-1]
            cov[int(item[1]), int(item[0])] = item[-2]+item[-1]
        return cov

    cov = readcov(np.loadtxt(params['base_dir']+params['cov_file']))
    if len(mask)!=len(cov):
        print("mask size not the same as cov, mask is fixed to match cov", flush=True)
        if len(mask)>len(cov):
            mask = mask[:len(cov)]
        else:
            masknew = np.zeros(len(cov))>1
            masknew[:len(mask)] = mask
            mask = masknew

    cov = cov[:,mask][mask,:]

    data = np.loadtxt(params['base_dir']+params['data_file'])[mask,1]
    inv_cov = np.linalg.inv(cov)
    sigma = np.sqrt(np.diag(cov))
   
    if pool is not None:
        if not pool.is_master():
            sys.stdout.flush()
            pool.wait()
            print("done", flush=True)
            print(" mpi = {0} done".format(MPI.COMM_WORLD.rank), flush=True)
            e = sys.exc_info()[0]
            print(e)
            print("#################")
            sys.exit(0)
    if pool is not None:
        if pool.is_master():
            if 'automaticgpu' in params:
                params['automaticgpu']['outdir'] = params['outdir']
                submitgpujob(params['automaticgpu'])
                gpunode = 'automaticgpu'
            ml_sampler_core(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, params['meanshiftArr'], params['stdshiftArr'], outdir, theory, priors, data, cov,  init, pool, nwalkers, device, dolog10index=[0,1], ypositive=False, temperatureArr=temperatureArr, omegab2cut=omegab2cut, docuda=False, tsize=tsize, gpunode=gpunode, nnmodel_in=nnmodel_in, params=params, method=method)
            end = time.time() 
            print("Runtime of the program is end - start", end-start)
            np.save(outdir+"/time.npy", end-start)
            if 'automaticgpu' in params:
                gpufile = os.path.join(outdir, "gpunodeinfo.pkl")
                with open(gpufile, 'rb') as f:
                    gpuinfo = pickle.load(f)
                jobid = gpuinfo["jobid"]
                os.system("scancel {0}".format(jobid))


    if pool is not None:
        pool.close()
        





if __name__=="__main__":
    main()
