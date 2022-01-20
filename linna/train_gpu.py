import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
from linna.predictor_gpu import * 
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from linna.nn import *
from scipy.special import erf
from scipy.stats import chi2
import io
from torch.utils import mkldnn as mkldnn_utils
try:
    import mpi4py.rc
    mpi4py.rc.initialize = False
except:
    print("no mpi")

if __name__=="__main__":
    if sys.argv[2] == "cuda":
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





