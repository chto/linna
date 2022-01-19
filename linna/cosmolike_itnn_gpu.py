import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
import sys
path = "/home/users/chto/code/lighthouse/analysis/"
sys.path.append(path+"/../python/")
sys.path.append(path+"/../python/datavector/")
sys.path.append(path+"/../lib/")
print(path)
import util_chto
import generate_data_vector
import  generate_batch_theory_datavector
from schwimmbad import MPIPool
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import sampler
import sys
import emcee
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from linna import *
import numpy as np 
import sys
sys.path.append("/home/users/chto/code/lighthouse/python/datavector/")
import util_chto
sys.path.append("/home/users/chto/code/lighthouse/python/")
#from cosmolike_libs_real_mpp_cluster import *
import matplotlib.pyplot as plt
from pathlib import Path
import run_4x2ptN_wrapper
import fitsio
import cosmolike_libs_real_mpp_cluster 
import seaborn as sns
import pandas as pd
import argparse
import generate_simulated_datavector
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
        #temp = np.array([0.305, 2.27E-9, 0.98, 0.0505, 1E-3, 0.733,  2.363, 2.285, 2.745, 2.093, 2.381, 0.02279545454545455, -0.025840909090909095, -0.009463636363636363, -0.038681818181818185, -0.04022727272727273, -0.042527272727272726, -0.06278636363636364, -0.03815, -0.12289999999999998, 0.05831363636363637, -0.05794090909090908, 0.047650000000000005, -0.10059545454545454, -0.05727272727272725, 4.104545454545454, 5.160909090909091, 0.9020454545454546, 0.5202727272727272, -0.1100000000000001, 1.1315454545454546, 0.26509090909090904])
        #temp_new = np.copy(temp)
        #temp_new[-6:-2] = x[1]
        #data = temp_new
        data = x[1]
        data_file = os.path.join(outdirs, "data_{0}".format(x[0]))
        print(data_file)
        outfile = bytes(data_file, encoding='utf-8')
        #mask = np.zeros(2712)
        #mask[1140:1152]=1
        #mask = mask>0
        if os.path.isfile(outfile):
            print("skip", outfile)
            return np.loadtxt(outfile.decode())[self.mask,1]
        #try:
        print("work on: {0}".format(x[0]))
        sys.stdout.flush()
        self.datavector_writer(data, outfile)
        #except:
        #    print("error on: {0}, mpi = {1}".format(x[0], MPI.COMM_WORLD.rank), flush=True)
        #    e = sys.exc_info()[0]
        #    print(e)
        #    print("#################")
        try: 
            data = np.loadtxt(outfile.decode())    
            if len(self.mask)>len(data):
                self.mask = self.mask[:len(data)]
            data = data[self.mask, 1]
        except:
            data = np.zeros_like(np.where(self.mask>0)[0])
        return data

def main():
    import time
    start = time.time()
    params = util_chto.chto_yamlload(sys.argv[3], parent_dir="/home/users/chto/code/lighthouse/analysis/yamlfiles/")         
   
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

    #init = np.array([4., 1.0, 0.2, 0])
    ntrainArr = params['ntrainArr']
    nvalArr =  params['nvalArr'] 
    nkeepArr =  params['nkeepArr']
    ntimesArr =  params['ntimesArr']
    ntautolArr =  params['ntautolArr']
    temperatureArr =   params['temperatureArr']
    nnmodel_in = eval(params['nnmodel'])
    #ntrainArr = [500, 500, 500]
    #nvalArr = [50, 50, 50]
    #nkeepArr = [5, 5,10]
    #ntimesArr = [10, 10, 50]
    #ntautolArr = [0.03, 0.03, 0.01] 
    #temperatureArr = [5.0, 3.0, 1.0]

    #mask = np.loadtxt("/oak/stanford/orgs/kipac/users/chto/lighthouse/analysis/backup/desy1/yamlfile/chains_cov_v8_multiplicative_bias/6x2pt+N_v1_weight_v2_omeganu_multinet_IA.yaml.mask")[:,1]
    #params['mask_file'] = "/home/users/chto/code/lighthouse//analysis/dataanalysis/desy1/yamlfile/chains_cov_v8_multiplicative_bias/6x2pt+N_v1_weight_v2_omeganu_multinet_IA.yaml.mask" 
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
                np.loadtxt(maskfile)
                break
            except:
                pass
    
    #params['statsnames']=['xip', 'xim', 'gammat', 'wtheta', 'w_cg', 'cluster_N', 'w_cc', 'gamma_c']
    init_cosmolike = cosmolike_libs_real_mpp_cluster.Initlized_cosmolike(bytes(maskfile, encoding='utf-8'), params)
    init_cosmolike.set_cosmolike()
    #cosmolike_libs_real_mpp_cluster.init_all_nuisance_param(params)
    priors, init = get_prior_dic_init(params)


    try:
        import torch.distributed as dist    
        import socket
        if size==1:
            assert(0)
        curr_env = os.environ.copy()
        tsize=40
        info = dict()
        if rank == 0:
            host = socket.gethostname()
            address = socket.gethostbyname(host)
            info.update(dict(MASTER_ADDR=address, MASTER_PORT='12345'))

        info = comm.bcast(info, root=0)
        info.update(dict(WORLD_SIZE=str(tsize), RANK=str(rank)))
        os.environ.update(info)
        #if rank<tsize:
        #    dist.init_process_group(backend='gloo',  rank=rank, world_size=tsize) 
        pool = chtoPool(comm)

    except:
        print("no MPI", flush=True)
        pool = None
        tsize=1

    mask = np.loadtxt(maskfile)[:,1]
    mask = mask>0
    if (pool is None) or pool.is_master():
            datavector_writer = generate_batch_theory_datavector.generate_base_file(params)
            #datavector_writer(init, bytes(os.path.join(outdir, "test.dat"), encoding='utf-8'))
            #params['statsnames']=['cluster_N', 'gamma_c']
            theory = Model_func(datavector_writer, mask)
   
        
    def readcov(covin):
        cov  = np.zeros((int(np.max(covin[:,0]))+1, int(np.max(covin[:,0]))+1))
        for item in covin:
            cov[int(item[0]), int(item[1])] = item[-2]+item[-1]
            cov[int(item[1]), int(item[0])] = item[-2]+item[-1]
        return cov
    cov = readcov(np.loadtxt(params['base_dir']+params['cov_file']))[:,mask][mask,:]
    data = np.loadtxt('/home/users/chto/code/lighthouse/analysis/speedup/ml/data/demo_data/'+"combined.txt")[mask,1]
    inv_cov = np.linalg.inv(cov)
    sigma = np.sqrt(np.diag(cov))
   
    #index = [0,1,2,3,4,5, 26,27,28,29] 
    ########################################
    if pool is not None:
        if not pool.is_master():
            pool.wait()
            print("done", flush=True)
            print(" mpi = {0} done".format(MPI.COMM_WORLD.rank), flush=True)
            e = sys.exc_info()[0]
            print(e)
            print("#################")
            sys.exit(0)
    #assert(0)
    ml_sampler(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, outdir, theory, priors, data, cov,  init, pool, nwalkers, device, dolog10index=[0,1], ypositive=False, temperatureArr=temperatureArr, omegab2cut=[3,5,0.005,0.039], docuda=False, tsize=tsize, gpunode=gpunode, nnmodel_in=nnmodel_in, params=params)
    end = time.time() 
    print("Runtime of the program is end - start", end-start)
    np.save(outdir+"/time.npy", end-start)


    if pool is not None:
        pool.close()
        

if __name__=="__main__":
#    import sys
#    import trace

# define Trace object: trace line numbers at runtime, exclude some modules
#    tracer = trace.Trace(
#        ignoredirs=[sys.prefix, sys.exec_prefix],
#        ignoremods=[
#            'inspect', 'contextlib', '_bootstrap',
#            '_weakrefset', 'abc', 'posixpath', 'genericpath', 'textwrap'
#        ],
#        trace=1,
#        count=0)

# by default trace goes to stdout
# redirect to a different file for each processes
#    sys.stdout = open('trace_{:04d}.txt'.format(MPI.COMM_WORLD.rank), 'w')
#    tracer.runfunc(main())
    main()
