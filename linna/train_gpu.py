import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
import sys
path = "/home/users/chto/code/lighthouse/analysis/"
sys.path.append(path+"/../python/")
sys.path.append(path+"/../python/datavector/")
sys.path.append(path+"/../lib/")
sys.path.append(path+"/../python/nnacc/nnacc/")
from predictor import * 
import util_chto
import generate_data_vector
import  generate_batch_theory_datavector
from schwimmbad import MPIPool
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import predictor
import sampler
import sys
import emcee
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from nn import *
from scipy.special import erf
from scipy.stats import chi2
import io
from torch.utils import mkldnn as mkldnn_utils
import mpi4py.rc
mpi4py.rc.initialize = False
import itnn_accelerator_batch_v2_gpu
import torch
print(path)

if __name__=="__main__":
    while(not torch.cuda.is_available()):
        print("no cuda", flush=True)
    outdir = sys.argv[1]
    f = open(outdir+"/model_pickle.pkl", 'rb')
    model = pickle.load(f) 
    f.close()
    f = open(outdir+"/model_args.pkl", 'rb')
    args = pickle.load(f) 
    f.close()
    model(*args)
    f = open(outdir+"/finish.pkl", 'wb')
    pickle.dump([True], f)
    f.close()





