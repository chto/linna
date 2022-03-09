import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    from schwimmbad import MPIPool
    nompi=False
except:
    nompi=True
    print("no mpi")
import glob
import pickle
from linna import predictor_gpu
from linna import sampler
import sys
import emcee
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from .nn import *
from scipy.special import erf
from scipy.stats import chi2
import io
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import mkldnn as mkldnn_utils
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tempfile
import numdifftools as nd
from scipy.stats import multivariate_normal

def makepositivedefinite(cov, fcut=0.99):
    eigvals, eigvec = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvec = eigvec[:,::-1]
    eigvals[eigvals<0]=0
    cumsum = np.cumsum(eigvals)
    plt.plot(cumsum)
    cumsum/=np.max(cumsum)
    ind = np.argmin(np.abs(cumsum-fcut))
    eigvals[ind:]=eigvals[ind]
    return eigvec @ np.diag(eigvals) @ eigvec.T

##Auxilery function 
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_good_walker_list(log_prob_samples):
    x = np.mean(log_prob_samples[-10000:,:], axis=0)
    X = np.array(list(zip(x,np.zeros(len(x)))), dtype=np.int)
    ms = KMeans()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    best = labels[np.argmax(cluster_centers[:,0])]
    print(np.where(labels==best)[0], cluster_centers, labels)
    return np.where(labels==best)[0] 

def read_chain_and_cut(chainname, nk, ntimes=20, walkercut=False, method="emcee", flat=False):
    if method=="emcee":
        reader = emcee.backends.HDFBackend(chainname, read_only=True)
    elif method =="zeus":
        reader = sampler.Zeusbackend(chainname)
    else:
        raise NotImplementedError(method)
    if nk > ntimes:
        print("Error: keep number greater then chain samples. nk: {0}, ntimes: {1}. This will lead to inclusion of all burn in step".format(nk, ntimes))
    if method =="zeus":
        nk = int(np.nanmedian(reader.get_autocorr_time())*nk)
    else:
        nk = int(np.median(reader.get_autocorr_time(quiet=True))*nk)
    chain = reader.get_value("chain_transformed", discard=0, flat=False, thin=1)
    log_prob_samples = reader.get_log_prob(discard=0, flat=False, thin=1)
    if walkercut:
        good_walker_list = get_good_walker_list(log_prob_samples)
    else:
        good_walker_list = np.arange(0, len(log_prob_samples[0]))
    chain = chain[int(-1*nk):,good_walker_list,:].reshape(-1, len(chain[0,0]))
    log_prob_samples = log_prob_samples[int(-1*nk):, good_walker_list]
    if flat:
        log_prob_samples = log_prob_samples.reshape(-1,1)
    return chain, log_prob_samples, reader

def _dummy_callback(x):
    pass 

if not nompi:
    class chtoPool(MPIPool):
        """
        A reimplimentation of ``schwimmbad.MPIPool`` that will not broacast redundant function
        """
        def __init__(self, comm=None):
            """
            Args:
                comm (mpi4py.comm): an MPI communicator
            """
            super(chtoPool, self).__init__(comm, use_dill=False)
            self.noduplicate=False
            self.worker_already_get_func_list=[]
        def wait(self):
            """
            Walkers will listen to the main process
            """
            if self.is_master():
                return

            worker = self.comm.rank
            status = MPI.Status()
            old_func=None
            while True:
                task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG,
                                      status=status)
                if task is None:
                    break

                #if self.rank==2:
                #    print("wait", self.rank, task)
                #    print("\n\n\n\n\n\n\n\n\n", flush=True)
                #    assert(0)
                if task is None:
                    #print("break\n\n\n\\n\n\n\n\n\n\n\n", flush=True)
                    continue
                    #break
                if task[0] == "bcast":
                    noreturn=True
                    task = task[1]
                    task[1] = self.rank
                else:
                    noreturn = False
                func, arg = task
                try:
                    if func == "noduplicate":
                        func = old_func
                    elif func == "reset":
                        old_func = None
                        continue
                    elif func.f.noduplicate and (old_func is not None):
                        func = old_func
                    else:
                        old_func = func
                except:
                    old_func=None
                    
                result = func(arg)
                if not noreturn:
                    self.comm.send(result, self.master, status.tag)
        def map(self, worker, tasks, callback=None):
            """Evaluate a function or callable on each task in parallel using MPI.

            The callable, ``worker``, is called on each element of the ``tasks``
            iterable. The results are returned in the expected order (symmetric with
            ``tasks``).

            Args:
                worker (callable):
                    A function or callable object that is executed on each element of
                    the specified ``tasks`` iterable. This object must be picklable
                    (i.e. it can't be a function scoped within a function or a
                    ``lambda`` function). This should accept a single positional
                    argument and return a single object.
                tasks (iterable):
                    A list or iterable of tasks. Each task can be itself an iterable
                    (e.g., tuple) of values or data to pass in to the worker function.
                callback (callable, optional):
                    An optional callback function (or callable) that is called with the
                    result from each worker run and is executed on the master process.
                    This is useful for, e.g., saving results to a file, since the
                    callback is only called on the master thread.

            Returns:
                list:
                    A list of results from the output of each ``worker()`` call.
            """

            # If not the master just wait for instructions.
            if not self.is_master():
                self.wait()
                return

            if callback is None:
                callback = _dummy_callback

            workerset = self.workers.copy()
            tasklist = [(tid, [worker, arg]) for tid, arg in enumerate(tasks)]
            resultlist = [None] * len(tasklist)
            pending = len(tasklist)

            while pending:
                if workerset and tasklist:
                    worker = workerset.pop()
                    taskid, task = tasklist.pop()
                    if self.noduplicate:
                        if worker in self.worker_already_get_func_list: 
                            task[0] = "noduplicate"
                        else:
                            self.worker_already_get_func_list.append(worker)
                            
                    self.comm.send(task, dest=worker, tag=taskid)

                if tasklist:
                    flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    if not flag:
                        continue
                else:
                    self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

                status = MPI.Status()
                result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                        status=status)
                worker = status.source
                taskid = status.tag

                callback(result)

                workerset.add(worker)
                resultlist[taskid] = result
                pending -= 1

            return resultlist
        def noduplicate_close(self):
            """
            Reset no duplicate function
            """
            self.worker_already_get_func_list=[]
            self.noduplicate = False
            workerset = self.workers.copy()
            for tid, workers in enumerate(workerset):
                self.comm.send(["reset", ["reset", None]], dest=workers, tag=tid)
        def bcast(self, worker, args, sizemax):
            """
            Broadcast function to all the workers:

            Args:    
                worker (callable): a function which you want to parallelize 
                args (list): list of things to be passed to worker
                sizemax (int): number of worker you wish to use 
                
            
            """
            workerset = self.workers.copy()
            for tid, workers in enumerate(workerset):
                if workers < sizemax:
                    self.comm.send(["bcast", [worker, args]], dest=workers, tag=tid)
            worker(0)

class chtoMultiprocessPool:
    """
        pool class if one wish to to multiprocess
    """
    def __init__(self, nwalker):
        """

        Args:
            nwalker (int): number of process
        """
        self.pool =  Pool(nwalker)

    def map(self, worker, tasks, callback=None):
        """
        Args:
            worker (function): function of worker 
            taskes (list of objects): lists of talks 
        Returns:
             list of objects
            
        """
        returned = self.pool.map(worker, tasks) 
        return returned
        
    def noduplicate_close(self):
        """
        close the pool
        """
        self.pool.close()

    def is_master(self):
        return True

def gauss2unif(x):
    """
        transform a guaaisan distributed random variable to a uniformly distributed variable

        Args:
            x (torch.tensor): input 
        Returns:
            torch.tensor: output
    """
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))

def invgauss2unif(x):
    """
        inverse transform a guaaisan distributed random variable to a uniformly distributed variable

        Args:
            x (torch.tensor): input 
        Returns:
            torch.tensor: output
    """
    return np.sqrt(2)*torch.erfinv(2*x-1)

class Transform:
    """
        Transform parameters so that all the prior is gaussian with zero mean and unit variance
    """
    def __init__(self, priors):
        """
        Args:
            priors (dict): a dictionary of ``2d list``. Each key can be either ``gauss`` indicating gaussian prior or ``flat`` indicating uniforma prior. For entries with key==``gauss``, the first item indicates the mean and the second item indicates the 1 sigma error. For entries with ``flat`` key, the first item indicates the lower limit and the second item indicates the upper limit.   
        """
        self.priors = priors
    def __call__(self, x, returnnumpy=True, inputnumpy=True):
        """
        Transform perameters so that all the prior is gaussian with zero mean and unit variance 
        
        Args:
            x (nd array or torch array): array of parameters 
            returnnumpy (bool) : If true, then the return value will be in ``numpy`` array. Otherwise, the return value will be in ``torch`` tensor 
            inputnumpy  (bool) : If true, the return value should be in ``numpy`` array. Otherwise, the return value should be in ``torch`` tensor 
        Returns:
            numpy array or torch tensor: depends on the input parameters ``returnnumpy`` 
        """
        transformed_x = []
        if inputnumpy:
            x=torch.from_numpy(x.astype(np.float32)).to('cpu').clone().requires_grad_()
        if (len(x.shape)<2):
            x = x.reshape(-1,len(x))
        for i, p in enumerate(self.priors):
            if p['dist'] == 'gauss':
                transformed_x.append(x[:, i] * p['arg2'] + p['arg1'])
            else:
                transformed_x.append(gauss2unif(x[:, i]) * (p['arg2'] - p['arg1']) + p['arg1'])
        if returnnumpy:
            return torch.stack(transformed_x).detach().numpy().T.squeeze()
        else:
            return torch.stack(transformed_x).T.squeeze()

class invTransform:
    """
        Inverse the ```Transform``` function. 
    """
    def __init__(self, priors):
        """
        Args:
            priors (dict): a dictionary of ``2d list``. Each key can be either ``gauss`` indicating gaussian prior or ``flat`` indicating uniforma prior. For entries with key==``gauss``, the first item indicates the mean and the second item indicates the 1 sigma error. For entries with ``flat`` key, the first item indicates the lower limit and the second item indicates the upper limit.   
        """
        self.priors = priors
    def __call__(self, x, returnnumpy=True, inputnumpy=True):
        """
        Args:
            x (nd array or torch array): array of parameters 
            returnnumpy (bool) : If true, then the return value will be in ``numpy`` array. Otherwise, the return value will be in ``torch`` tensor 
            inputnumpy  (bool) : If true, the return value should be in ``numpy`` array. Otherwise, the return value should be in ``torch`` tensor 
        Returns:
            numpy array or torch tensor: depends on the input parameters ``returnnumpy`` 
        """
        transformed_x = []
        if inputnumpy:
            x=torch.from_numpy(x.astype(np.float32)).to('cpu').clone().requires_grad_()
        if (len(x.shape)<2):
            x = x.reshape(-1,len(x))
        for i, p in enumerate(self.priors):
            if p['dist'] == 'gauss':
                transformed_x.append((x[:, i]-p['arg1'])/p['arg2'])
            else:
                transformed_x.append(invgauss2unif((x[:, i]-p['arg1'])/ (p['arg2'] - p['arg1'])))
        if returnnumpy:
            return torch.stack(transformed_x).detach().numpy().T.squeeze()
        else:
            return torch.stack(transformed_x).T.squeeze()

class ArrayDataset(Dataset):
    """
        prepare data for torch
    """
    def __init__(self, X, y):
        """
        Args:
            X (nd array): numpy array
            y (nd array): numpy array
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i,:], self.y[i,:]

class Y_transform_data:
    """
    Transform data vector from y-->y/sigma
    """
    def __init__(self, sigma, device):
        """
        Args:
            sigma (float): sigma 
            device(string): "cpu" or "cuda"

        """
        self.device= device
        self.sigma = torch.from_numpy(sigma.astype(np.float32)).to(device).clone().requires_grad_()
    
    def __call__(self, y):
        """
        Args:
            y (torch tensor): data vector
        Returns:
            torch tensor: y/sigma
        """
        return y/self.sigma[None,:]
    
    def pickle(self, path):
        """
        Pickle the transform

        Args:
            path (string): name of the pickle file
        """
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)

    def transform_cov(self, cov):
        """
        Transform the associated covariance matrix if one transform the data vector by 1/sigma

        Args:
            cov (2d array): covariance matrix

        Returns:
            torch(2d array): transformed covariance matrix
        """
        return torch.diag(1/self.sigma.type(torch.float64)).inner(cov).inner(torch.diag(1/self.sigma.type(torch.float64)))

class Y_invtransform_data:
    """
    Transform data vector from y-->y sigma (Api is the same as ``Y_transform_data``)
    """
    def __init__(self, sigma, device):
        self.sigma = torch.from_numpy(sigma.astype(np.float32)).to(device).clone().requires_grad_()
        self.device= device
    
    def __call__(self, y):
        return y*self.sigma[None,:]
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)
            
class X_transform_class:
    """
    Transform parameters from x --> (x-mean)/std or x --> (log10(x)-mean)/std
    """
    def __init__(self, X_mean, X_std, device, dolog10index=None):
        """
        Args: 
            X_mean (torch tensor): mean of data vector 
            X_std  (torch tensor): std of data vector 
            device (string): "cpu" or "cuda"
            dolog10index (list, optional): which index you wish to have an additional log10 transform
        """
        self.X_mean = X_mean
        self.X_std = X_std
        self.dev = device
        self.dolog10index = dolog10index 
        
    def __call__(self, X):
        """
        Args:
            X (torch tensor): array with first dimension the same as the length of X_mean 
        Returns:
            torch tensor 
        """
        X1 = X.clone()
        if self.dolog10index is not None:
            for ind in self.dolog10index:
                if len(X1.shape)>1:
                    X1[:,ind] = torch.log10(X[:,ind])
                else:
                    X1[ind] = torch.log10(X1[ind])
        return (X1 - self.X_mean[None,:].to(self.dev)) / self.X_std[None,:].to(self.dev)
    
    def pickle(self, path):
        """
        Pickle the transform

        Args:
            path (string): name of the pickle file
        """
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)
                        
class Y_transform_class:
    """
    Transform data vector: y-->y*std+mean or np.exp(y-->y*std+mean)
    """
    def __init__(self, y_mean, y_std, dev, ypositive=False):
        """
        Transform data vector: y-->y*std+mean or np.exp(y-->y*std+mean)

        Args:
            y_mean (torch tensor): mean
            y_std (torch tensor): standard deviation 
            dev (string): cuda or cpu
            ypositive (bool): whether assume all data vector is possive. If true, we do np.exp(y-->y*std+mean). If false, we do y*std+mean 
        """

        self.y_mean = y_mean
        self.y_std = y_std
        self.dev = dev
        self.ypositive = ypositive
        
    def __call__(self, y):
        """
        Args:
            y (torch tensor): data 
        Returns:
            torch tensor: transformed data 
        """
        if self.ypositive:
            return torch.exp(y * self.y_std[None,:].to(self.dev) + self.y_mean[None,:].to(self.dev))
        else:
            return y * self.y_std[None,:].to(self.dev) + self.y_mean[None,:].to(self.dev)
    
    def pickle(self, path):
        """
        Pickle the transform

        Args:
            path (string): name of the pickle file
        """
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)

class Y_invtransform_class:
    """
    Transform data vector: y-->(y-mean)/std or (np.log(y)-mean)/std  (API tis the same as `` Y_transform_class``
    """
    def __init__(self, y_mean, y_std, data_tensor, dev, ypositive=False):
        self.y_mean = y_mean
        self.y_std = y_std
        self.dev = dev
        self.ypositive = ypositive
        self.data_tensor = data_tensor
        
    def __call__(self, y):
        if self.ypositive:
            return (torch.log(y)-self.y_mean[None,:].to(self.dev))/self.y_std[None,:].to(self.dev) 
        else:
            return (y-self.y_mean[None,:].to(self.dev))/self.y_std[None,:].to(self.dev)
    def transform_cov(self, cov):
        """
        Transform the associated covariance matrix if one transform the data vector by 1/sigma

        Args:
            cov (2d array): covariance matrix

        Returns:
            torch(2d array): transformed covariance matrix
        """
        if self.ypositive:
           expected = self.data_tensor 
           cov0 = (torch.diag(1/expected.type(torch.float64)).inner(cov).inner(torch.diag(1/expected.type(torch.float64)))).to(self.dev)
           cov0[cov0<=-1] = 1E-10-1
           cov1 = torch.log(1+cov0)
           cov2 =  (torch.diag(1/self.y_std.type(torch.float64)).inner(cov1).inner(torch.diag(1/self.y_std.type(torch.float64)))).to(self.dev)
           return cov2
        else:
           return (torch.diag(1/self.y_std.type(torch.float64)).inner(cov).inner(torch.diag(1/self.y_std.type(torch.float64)))).to(self.dev)
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)

class _FunctionWrapper(object):
    """
        Only for internal use
        :meta private:
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)

def retrieve_model(outdir, inshape, outshape, nnmodel_in=ChtoModelv2):
    """
    Retrieve the trained model

    Args:
        Outdir (string): directory of the outdir
        inshape (int): input vector size of the model 
        outshape (int): output vector size of the model 
        nnmodel_in (callable, optional): neural network instance defined in nn.py

    Returns:
        linna.predictor_gpu.Predictor: model
        linna.util.Y_invtransform_data: callable that transform the model to the same space as data vector  
        
    """
    ####Retrive the model 
    with open(os.path.join(outdir,'y_invtransform_data.pkl'), 'rb') as f:
        y_invtransform_data = CPU_Unpickler(f).load()
    with open(os.path.join(outdir,'X_transform.pkl'), 'rb') as f:
        X_transform = CPU_Unpickler(f).load()
    X_transform.dev = "cpu"
    with open(os.path.join(outdir,'y_transform.pkl'), 'rb') as f:
        y_transform = CPU_Unpickler(f).load()
    linearmodel = None
    y_transform.dev = "cpu"
    nnmodel = nnmodel_in(inshape, outshape, linearmodel)
    model = predictor_gpu.Predictor(inshape, outshape, X_transform=X_transform, y_transform=y_transform,device='cpu', outdir=outdir, model = nnmodel)
    model.load_checkpoint()
    return model, y_invtransform_data

def retrieve_model_wrapper_in(outdir, nnmodel_in=ChtoModelv2, no_grad=True):
    """
    Retrieve the trained model (more user friendly than `retrieve_model`)

    Args:
        Outdir (string): directory of the outdir
        nnmodel_in (callable, optional): neural network instance defined in nn.py
        no_grad (bool): True: not keep gradient information, Flase: keep gradient information

    Returns:
        model (callable): a function takes in cosmological and nuisance parameters (torch tensor) and returns the prediction of the data vector using the neural network. Note that the output is in the format of torch.tensor, so that its differentiation can be evaluated.

    """
    nshapein = np.loadtxt(os.path.join(outdir, "train_samples_x.txt")).shape[1]
    nshapeout = np.load(os.path.join(outdir, "train_samples_y.npy")).shape[1]
    model, y_invtransform_data = retrieve_model(outdir, nshapein, nshapeout, nnmodel_in=ChtoModelv2)
    if no_grad:
        model.model = model.model.to(memory_format=torch.channels_last)
        model.MKLDNN=True
    return lambda x: y_invtransform_data(model.predict(x, no_grad=no_grad).to("cpu").clone())

class NN_samplerv1:
    """
    A class to perform neural network sampling for each iteration 
    """
    def __init__(self, outdir, prior_range):
        """
        Args:
            outdir (str): base directory of output training and mcmc files
            prior_range (ndarray): 2d array. Each row represents the upper and lower limits of the prior 
        """
        self.outdir = outdir 
        self.prior_range = prior_range
        self.seed=123456 #random seed of training data generation 

    def generate_training_data(self, samples, model, pool=None, args=None, kwargs=None):
        """
        
        Generate predicted data vector from a set of parameters 

        Args:
            samples (ndarray): 2d array containing data with float type. Set of parameters in each row 
            model (function): a function that take a row of samples, args and kwargs and return the predicted data vector  
            pool (mpi pool, optional): a mpi pool instance that can do pool.map(function, iterables). 
            args, kwargs (lists, optional): args or kwargs to be passed to model
        Returns:
            ndarray: each row corresponds to the output of model at each row of the samples.
        """
        m = _FunctionWrapper(model, args, kwargs)
        filelist = glob.glob(os.path.join(args[0]+"/", "*"))
        for f in filelist:
            os.remove(f)
        if pool is not None:
            returned =  np.array(list(pool.map(m, samples)))
        else:
            returned =  np.array(list(map(m, samples)))
        filelist = glob.glob(os.path.join(args[0]+"/", "*"))
        for f in filelist:
            os.remove(f)
        return returned
    def gensample_flat(self, Nsamples, omegab2cut=None):
        """

        Generate parameters for training and validation using latin hypercube.

        Args:
            Nsamples (int): number of samples to be generated. 
            omegab2cut (list of int): 2 elements containing the lower and upper limits of omegab*h^2  
        Returns:
            ndarray: a 2d array containing data with float type. Parameters for training and validation  
        """
        samples=[]
        Nsample_in = Nsamples
        shiftAs=False
        while(len(samples)<Nsamples):
            samples = pyDOE2.lhs(len(self.prior_range), samples=int(Nsample_in), criterion="center",
                                  iterations=5, random_state=self.seed)
            samples-= 0.5
            samples*=2 
            for ind, prior in enumerate(self.prior_range):
                if (ind ==1)&(self.prior_range[1][1]<1E-5):
                    prior = np.log(prior)
                    shiftAs=True
                scaled = (prior[1]-prior[0])/2
                mean = (prior[1]+prior[0])/2
                samples[:, ind] = samples[:, ind]*scaled+mean
                if shiftAs:
                    if ind==1:
                        samples[:, ind] = np.exp(samples[:, ind])
            if omegab2cut is not None:
                ombh2 = samples[:,omegab2cut[0]]*samples[:, omegab2cut[1]]**2
                keep = (ombh2>omegab2cut[2])&(ombh2<omegab2cut[3])
                samples=samples[keep]
            Nsample_in+=1000
        return samples[:Nsamples]

    def gensample_chain(self, Nsamples, chain_in, nsigma, omegab2cut=None):
        """

        Generate parameters for training and validation from a chain using latin hyper cube.

        Args:
            Nsamples (int): number of samples to be generated. 
            chain_in (ndarray): a mcmc chain.
            nsigma (int): up to this number an mcmc chain is generated 
            omegab2cut (list of int): 2 elements containing the lower and upper limits of omegab*h^2  
        Returns:
            ndarray: a 2d array containing data with float type. Parameters for training and validation  
        """

        chain = deepcopy(chain_in)
        prior_in = deepcopy(self.prior_range)
        Nsamples=int(Nsamples)
        total_sample=0
        n_factor=1
        shiftAs=False
        if prior_in[1][1]<1E-5:
            shiftAs=True
            chain[:,1] = np.log(1e10*chain[:,1])
            prior_in[1][0] = np.log(1e10*prior_in[1][0])
            prior_in[1][1] = np.log(1e10*prior_in[1][1])
        Generator = sg.SampleGenerator(chain=chain, scale=nsigma)
        Generator.set_seed(self.seed)
        while(total_sample<Nsamples):
            x = Generator.get_samples(int(n_factor*Nsamples), "LH")
            if omegab2cut is not None:
                ombh2 = x[:,omegab2cut[0]]*x[:, omegab2cut[1]]**2
                keep = (ombh2>omegab2cut[2])&(ombh2<omegab2cut[3])
                x=x[keep]
            for i in range(x.shape[1]):
                keep = (x[:, i]>prior_in[i][0])&(x[:, i]<prior_in[i][1])
                x=x[keep]
            if shiftAs:
                x[:,1] = np.exp(x[:,1])/1E10
            n_factor+=1
            total_sample=x.shape[0]
        return x[:Nsamples]


    def gensample_chain_randomsample(self, Nsamples, chain_in, nsigma, omegab2cut=None):
        """

        Generate parameters for training and validation from a chain using latin hyper cube.

        Args:
            Nsamples (int): number of samples to be generated. 
            chain_in (ndarray): a mcmc chain.
            nsigma (int): up to this number an mcmc chain is generated 
            omegab2cut (list of int): 2 elements containing the lower and upper limits of omegab*h^2  
        Returns:
            ndarray: a 2d array containing data with float type. Parameters for training and validation  
        """

        chain = deepcopy(chain_in)
        prior_in = deepcopy(self.prior_range)
        Nsamples=int(Nsamples)
        total_sample=0

        if omegab2cut is not None:
            ombh2 = chain[:,omegab2cut[0]]*chain[:, omegab2cut[1]]**2
            keep = (ombh2>omegab2cut[2])&(ombh2<omegab2cut[3])
            chain=chain[keep]
        for i in range(chain.shape[1]):
            keep = (chain[:, i]>prior_in[i][0])&(chain[:, i]<prior_in[i][1])
            chain=chain[keep]
        np.random.seed(self.seed)
        return chain[np.random.randint(0, len(chain), Nsamples)]

    def emcee_sample(self, log_prob, ndim, nwalkers, init, pool, transform, ntimes=50, tautol=0.01, dlnp=None, ddlnp=None, meanshift=0.1, stdshift=0.1, nk=1):
        """

        Generate MCMC chains using emcee.

        Args:
            log_prob (function): function of posterior. 
            ndim (int): the dimension of posterior 
            nwalkers (int): number of mcmc walkers 
            init (ndarray): array of init points of the sampler 
            pool (mpi pool, optional): a mpi pool instance that can do pool.map(function, iterables)
            transform (function): mapping mcmc samples to actually parameters 
        """


        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform)
        if (ddlnp is not None)&(dlnp is not None):
            samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, 0, 0, outdir=self.outdir, overwrite=False, ntimes=ntimes, method = "emcee", incremental=True, progress=False, tautol=tautol, meanshift=meanshift, stdshift=stdshift, nk=nk)

    def Zeus_sample(self, log_prob, ndim, nwalkers, init, pool, transform, ntimes=50, tautol=0.01, dlnp=None, ddlnp=None, meanshift=0.1, stdshift=0.1, nk=1):
        """

        Generate MCMC chains using zeus.

        Args:
            log_prob (function): function of posterior. 
            ndim (int): the dimension of posterior 
            nwalkers (int): number of mcmc walkers 
            init (ndarray): array of init points of the sampler 
            pool (mpi pool, optional): a mpi pool instance that can do pool.map(function, iterables)
            transform (function): mapping mcmc samples to actually parameters 
        """


        max_n = 1000000
        x0 = init+0.01*np.random.randn(nwalkers, ndim)
        samp = sampler.ZeusSampler(log_prob, ndim, nwalkers, x0=x0, transform=transform)
        samp.sample(pool, max_n, outdir=self.outdir, overwrite=False, ntimes=ntimes, incremental=True, progress=False, tautol=tautol, meanshift=meanshift, stdshift=stdshift, nk=nk)
    def _HMC_sample(self, log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, transform, samp_steps, samp_eps):
        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform)
        samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, samp_steps, samp_eps, outdir=self.outdir, overwrite=True, ntimes=50, method = "hmc", incremental=True, progress=True)
    def _NUTS_sample(self, log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, transform,  Madapt):
        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform, torchspeed=True)
        samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, 0,0,Madapt, outdir=self.outdir, overwrite=False, ntimes=50, method="nuts", incremental=True, progress=True)

def gaussianlogliklihood( m, data, invcov):
    d = m-data
    return (d@(invcov)@d.T*(-0.5))[0][0]

class Log_prob:
    """
    Class do loglikelihood
    """
    def __init__(self, data_new, invcov_new, model, y_invtransform_data, transform, temperature, loglikelihoodfunc, nograd=False):
        """
        Args:
            data_new (array or torch tensor): data vector
            invcov_new (array or torch tensor): inverse of the covariance matrix 
            model (linna.predictor_gpu.Predictor): model
            y_invtransform_data (linna.util.Y_invtransform_data): data transform
            transform (``Transform``): transform parameter
            temperature (float): inflat the likelihood by L--> L/T
            loglikelihoodfunc (callable): function of model, data , inverse of covariance matrix and return the log liklihood value 
            nograd (bool): whether to retain gradiant information 
        """
        if not torch.is_tensor(data_new):
            self.data_new = torch.from_numpy(data_new.astype(np.float32)).to("cpu").clone().requires_grad_()
        else:
            self.data_new = data_new 
        if not torch.is_tensor(invcov_new):
            self.invcov_new = torch.from_numpy(invcov_new.astype(np.float32)).to("cpu").clone().requires_grad_()
        else:
            self.invcov_new = invcov_new
        self.model = model
        self.y_invtransform_data = y_invtransform_data
        self.transform = transform
        self.T = temperature
        self.no_grad = nograd
        self.loglikelihoodfunc = loglikelihoodfunc
        self.noduplicate=True

    def __call__(self, x, returntorch=True, inputnumpy=True):
        """
        Args:
            x (numpy array or tensor):
            returntorch (bool): whether to return torch tensor or numpy array
            inputnumpy (bool): whether the input is in numpy array or torch tensor
        
        Returns:
            tensor or numpy array

        """
        if not torch.is_tensor(x):
            x=torch.from_numpy(x.astype(np.float32)).to("cpu").clone().requires_grad_()
        x_in  = self.transform(x,inputnumpy=False, returnnumpy=False)
        m = self.y_invtransform_data(self.model.predict(x_in,no_grad=self.no_grad))
        #d = m-self.data_new
        #(d@(self.invcov_new)@d.T*(-0.5))/self.T+lnprior(x)

        like = self.loglikelihoodfunc(m, self.data_new, self.invcov_new)/self.T + lnprior(x) #(d@(self.invcov_new)@d.T*(-0.5))/self.T+lnprior(x)
        if torch.isnan(like): 
            return -torch.inf
        else:
            if returntorch:
                return like
            else:
                return like.detach().numpy() 
   
class Dlnp:
    """
    Class do derivative of loglikelihood (API the same as ``Log_prob``)
    """
    def __init__(self, data_new, invcov_new, model, y_invtransform_data, transform, temperature):
        self.log_prob = Log_prob(data_new, invcov_new, model, y_invtransform_data, transform, temperature)
    def __call__(self, x, lnP=None, returntorch=None, inputnumpy=None):
        if not torch.is_tensor(x):
            x=torch.from_numpy(x.astype(np.float32)).to("cpu").clone().requires_grad_()
        if lnP is None:
            lnP = self.log_prob(x, returntorch=True, inputnumpy=False)
        grad = torch.autograd.grad(lnP, x)[0]
        return grad.detach().numpy() 

class Ddlnp:
    """
    Class do 2nd derivative of loglikelihood (API the same as ``Log_prob``)
    """
    def __init__(self, data_new, invcov_new, model, y_invtransform_data, transform, temperature):
        self.log_prob = Log_prob(data_new, invcov_new, model, y_invtransform_data, transform, temperature)
    def __call__(self, x):
        x=torch.from_numpy(x.astype(np.float32)).to("cpu").clone().requires_grad_()
        lnp = self.log_prob(x, returntorch=True, inputnumpy=False)
        grad = torch.autograd.grad(lnp, x, create_graph=True)[0]
        hess=[]
        for i in range(len(grad)):
            hess.append(torch.autograd.grad(grad[i], x, retain_graph=True)[0])
        hess = torch.stack(hess)
        return hess.detach().numpy()



class Auxilleryfunc:
    """
    Class for internal use 
    :meta private:
    """
    def __init__(self, data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device):
        self.inv_cov_tensor = inv_cov_tensor
        self.transformed_cov = cov_tensor 
        self.transformed_cov = y_inv_transform.transform_cov(y_transform_data.transform_cov(self.transformed_cov))
        self.inv_transformed_cov = torch.inverse(self.transformed_cov).type(torch.float32).detach()
        self.y_transform_data=y_transform_data
        self.y_inv_transform = y_inv_transform
        self.device = device
        self.data = data_in
        self.data_in = torch.nan_to_num(self.y_inv_transform(self.y_transform_data(self.data)).to(self.device), nan=1E-30).detach()
    def __call__(self, y_pred, y_target):
            y_target_in = self.y_inv_transform(self.y_transform_data(y_target)).detach()
            mask = (y_target==1E-30) | (y_target==1E10) | (self.data_in==1E-30)
            notmask = torch.logical_not(mask) 
            y_pred_in = y_pred
            delta = (y_pred_in - self.data_in)
            delta[mask] = 0
            chisqnnd = torch.sum(((delta @  self.inv_transformed_cov) * delta), dim=-1)
            
            delta = (y_target_in - self.data_in)
            delta[mask] = 0
            chisqMd = torch.sum(((delta @  self.inv_transformed_cov) * delta), dim=-1)
         
            delta = y_target_in - y_pred_in
            delta[mask] = 0
            chisqMnn = torch.sum(((delta @  self.inv_transformed_cov) * delta), dim=-1)
            chisqMd[chisqMd<0.5*len(y_target[0])] =  0.5*len(y_target[0])
            loss = chisqMnn/chisqMd
            return loss, chisqMd, chisqnnd

class Loss_fn:
    """
    Class defined loss function 
    """
    def __init__(self, data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device):
        """
        Args:
            data_in (torch tensor): data vector
            cov_tensor (torch tensor): covariance matrix 
            inv_cov_tensor (torch tensor): inverse of the covariance matrix 
            y_transform_data (linna.util.Y_transform_data): data transform
            y_inv_transform (``Y_invtransform_class``):
            device (string): cpu or cuda
        """
        self.auxileryfunction = Auxilleryfunc(data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)
    def __call__(self, y_pred, y_target):
        """
        Args:
            y_pred (tensor): predicted data vector by nn
            y_target (tensor): targeted data vector

        Return:
            torch.float: loss 
        """
        loss, _, _ = self.auxileryfunction(y_pred, y_target)
        loss = torch.mean(loss)
        return loss

class Val_metric_fn:
    """
    Class for validation metric (API the same as loss)
    """
    def __init__(self, data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device):
        self.auxileryfunction = Auxilleryfunc(data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)
    def __call__(self, y_pred,y_target):
        loss, chisqMd, chisqnnd = self.auxileryfunction(y_pred,y_target)
        fracerr = torch.abs((chisqnnd)/chisqMd-1)
        return torch.tensor([torch.median(loss),torch.max(fracerr) ,torch.median(fracerr)])

class LogPrior:
    """
    Priors handling
    """
    def __init__(self, prior):
        """
        Args:
            prior (list of dict): each entry is ['dist': [flat or gauss], 'arg1': lower bound for flat or mean for gauss, 'arg2': upper bound for flat or std for gauss]
        """
        self.prior = prior 
    def __call__(self, xlist):
        """
        Args:
            xlist (list): data 

        Returns:
            float: prior
        """
        logp = 0
        for ind, x in enumerate(xlist):
            item = self.prior[ind]
            if item['dist'] == 'flat':
                if x < item['arg1']:
                    return -np.inf
                if x > item['arg2']:
                    return -np.inf
            if item['dist'] == 'gauss':
                logp += -0.5*(x-item['arg1'])**2/item['arg2']**2
        return logp


def lnprior(x):
    """
    internal function
    :meta private:
    """
    return -0.5 * torch.sum(x.square())

def generate_training_point(theory, nnsampler, pool, outdir, ntrain, nval, data, invcov, chain=None, nsigma=1, omegab2cut=None, options=0, negloglike=None, nbest_in=None, chisqcut=None):
    """
    Generate training point 

    Args:
        theory (callable): model  
        nnsampler (``NN_samplerv1`` instance):
        pool (``chtoPool`` instance): mpi pool
        outdir (string): output directory 
        ntrain (int): number of training data
        nval (int): number of validation data
        data (1d array): float array, data vector 
        invcov (2d array): float array, inverse of covariance matrix
        chain (array or None): if None: generate from prior, if and array: training sample will be generated using thie chain 
        nsigma (int): if option ==0, this means we build a LH in nsignma region of the chain. 
        omegab2cut (list or None): additional cut on omegabh2, if list, omegab2cut = [index of omegab, index of h, lower limit, upper limit]
        options (int): if 0, generate using Latin Hypercube. If 1, random sample the chain
        negloglike (callable): if not None, generate one sample usin maximum likelihood method (minimize negloglike)
        nbest_in (int, optional): number of samples to be included in the training set according to the optimizer 
        chisqcut (float, optional): cut the training data if there chisq is greater than this value 
    """
    if (pool is None) or pool.is_master():
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if not os.path.isfile(os.path.join(outdir, "train_samples_x.txt")):
            if chain is None:
                samples = nnsampler.gensample_flat(ntrain, omegab2cut=omegab2cut)
            else:
                if options==0:
                    samples = nnsampler.gensample_chain(ntrain, chain, nsigma, omegab2cut=omegab2cut)
                elif options==1:
                    samples = nnsampler.gensample_chain_randomsample(ntrain, chain, nsigma, omegab2cut=omegab2cut)
                else:
                    print("options : {0} not recognized".format(options))
                    assert(0)

            np.savetxt(os.path.join(outdir, "train_samples_x.txt"), samples)
        if not os.path.isfile(os.path.join(outdir, "val_samples_x.txt")):
            if chain is None:
                samples = nnsampler.gensample_flat(nval, omegab2cut=omegab2cut)
            else:
                if options==0:
                    samples = nnsampler.gensample_chain(nval, chain, nsigma, omegab2cut=omegab2cut)
                elif options==1:
                    samples = nnsampler.gensample_chain_randomsample(nval, chain, nsigma, omegab2cut=omegab2cut)
                else:
                    print("options : {0} not recognized".format(options))
                    assert(0)

            np.savetxt(os.path.join(outdir, "val_samples_x.txt"), samples)
        outtrain = os.path.join(outdir, "train/")
        outval = os.path.join(outdir, "val/")
        if not os.path.isdir(outtrain):
            os.makedirs(outtrain)
        if not os.path.isdir(outval):
            os.makedirs(outval)
        #generate training data
        if not os.path.isfile(os.path.join(outdir, "train_samples_y.npy")):
            train_x = np.loadtxt(os.path.join(outdir, "train_samples_x.txt"))#[[1430,1431],:]
            train_y = nnsampler.generate_training_data(zip(range(len(train_x)), train_x), theory, pool=pool, args=[outtrain])
            np.save(outdir+"train_samples_y.npy", train_y)

        #generate validation data
        if not os.path.isfile(os.path.join(outdir, "val_samples_y.npy")):
            val_x = np.loadtxt(os.path.join(outdir, "val_samples_x.txt"))
            val_y = nnsampler.generate_training_data(zip(range(len(val_x)), val_x), theory, pool=pool, args=[outval])
            np.save(outdir+"val_samples_y.npy", val_y)
        #Generate best point
        if negloglike is not None:
            if not os.path.isfile(os.path.join(outdir, "best_samples_x.txt")):
                train_x = np.loadtxt(os.path.join(outdir, "train_samples_x.txt"))
                bestx_mean = minimize(negloglike, train_x[0], method='Nelder-Mead', tol=1e-6).x
                invHess = np.linalg.inv(makepositivedefinite(nd.Hessian(negloglike)(bestx_mean))) 
                bestx = multivariate_normal.rvs(mean=bestx_mean, cov=invHess, size=nbest_in, random_state=None)
                np.savetxt(os.path.join(outdir, "best_samples_x.txt"), bestx)
                bestx_val = multivariate_normal.rvs(mean=bestx_mean, cov=invHess, size=int(nbest_in/ntrain*nval), random_state=None)
                np.savetxt(os.path.join(outdir, "best_samples_x_val.txt"), bestx_val)
            if not os.path.isfile(os.path.join(outdir, "best_samples_y.npy")):
                bestx = np.loadtxt(os.path.join(outdir, "best_samples_x.txt"))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    besty = nnsampler.generate_training_data(zip(range(len(bestx)), bestx), theory, pool=pool, args=[tmpdirname])
                np.save(outdir+"best_samples_y.npy", besty)
                bestx = np.loadtxt(os.path.join(outdir, "best_samples_x_val.txt"))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    besty = nnsampler.generate_training_data(zip(range(len(bestx)), bestx), theory, pool=pool, args=[tmpdirname])
                np.save(outdir+"best_samples_y_val.npy", besty)
        if chisqcut is not None:
            chisqcut_all(data, invcov, chisqcut, os.path.join(outdir, "train_samples_y.npy"), os.path.join(outdir, "train_samples_x.txt"))
            chisqcut_all(data, invcov, chisqcut, os.path.join(outdir, "val_samples_y.npy"), os.path.join(outdir, "val_samples_x.txt"))
            if negloglike is not None:
                chisqcut_all(data, invcov, chisqcut, os.path.join(outdir, "best_samples_y.npy"), os.path.join(outdir, "best_samples_x.txt"))
                chisqcut_all(data, invcov, chisqcut, os.path.join(outdir, "best_samples_y_val.npy"), os.path.join(outdir, "best_samples_x_val.txt"))

def chisqcut_all(data, invcov, chisqcut, fnamey, fnamex):
    """
        Internal function
    """
    y = np.load(fnamey)
    x = np.loadtxt(fnamex)
    chisq = np.array([y_.dot(invcov).dot(y_) for y_ in y])
    y= y[chisq<chisqcut]
    x = x[chisq<chisqcut]
    np.save(fnamey, y)
    np.savetxt(fnamex, x)

def train_nn(outdir, model, train_x, train_y, val_x, val_y, X_transform, y_transform, loss_fn, val_metric_fn,dev = "cpu", verbose=False, retrain=True, pool=None, nocpu=False, size=0, rank=0, params=None):
    """
        Internal function
    """
    if not retrain:
        if os.path.isfile(os.path.join(outdir, "best.pth.tar")):
            return
    model = predictor_gpu.Predictor(train_x.shape[-1], train_y.shape[-1], X_transform=X_transform, y_transform=y_transform,device=dev, optim="automatic", model=model, scheduler=None, outdir=outdir)
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    train_dataset = ArrayDataset(train_x, train_y)
    val_dataset = ArrayDataset(val_x, val_y)
    sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_y))
    train_loss, val_metric = model.train(train_loader, num_epochs, loss_fn, val_loader, val_metric_fn, initfrombest=True, pool=None, nocpu=nocpu, rank=rank, size=size)
    if verbose and (rank==0):
        fig, axes = plt.subplots(1,4,figsize=(15,5))
        axes[0].plot(np.arange(1, len(train_loss)+1), train_loss, label='Training Mean')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[1].plot(np.arange(1, len(val_metric[:,0])+1), val_metric[:,0], label='Validation loss')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[2].plot(np.arange(1, len(val_metric[:,0])+1), val_metric[:,1], label='error max')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[3].plot(np.arange(1, len(val_metric[:,0])+1), val_metric[:,2], label='error median')
        axes[3].set_yscale('log')
        axes[3].legend()
        plt.xlabel('Epoch')
        plt.ylabel('$\\chi^2 / dof$')
        plt.legend()
        plt.savefig(os.path.join(outdir, "trainniing.png"))
    return model

def median_absolute_deviation(y, median, dim):
    """
        Internal function
    """
    df = torch.abs(y-median)
    return df.median(axis=dim).values

def train_NN( nnsampler, cov, inv_cov, sigma, outdir_in, outdir_list,data, dolog10index=None, ypositive=False, retrain=True, norder=2, temperature=None, docuda=False, pool=None, tsize=1, nnmodel_in=None, params=None, usebest=False):
        """
        Internal function
        """
        #TrainNN
        if docuda:
            device = "cuda"
        else:
            device = "cpu"

        ########################################
        inv_cov_tensor = torch.tensor(inv_cov, dtype=torch.float64).to(device)
        cov_tensor = torch.tensor(cov, dtype=torch.float64).to(device)
        y_transform_data = Y_transform_data(sigma, device=device)
        y_transform_data.pickle(os.path.join(outdir_in,'y_transform_data.pkl'))
        y_invtransform_data = Y_invtransform_data(sigma, device=device)
        y_invtransform_data.pickle(os.path.join(outdir_in,'y_invtransform_data.pkl'))
        data_tensor = torch.from_numpy(data.astype(np.float32)).to(device).clone().requires_grad_() 

        def XT1(X):
            X1= torch.tensor(X,dtype=torch.float32)
            if dolog10index is not None:
                for ind in dolog10index:
                    X1[:,ind] = torch.log10(X1[:,ind])
            return X1


        train_x = []
        train_y = []
        val_x = []
        val_y = []
        for outdir in outdir_list:
            _ = np.loadtxt(os.path.join(outdir, "train_samples_x.txt"))
            if len(_)>1:
                train_x.append(_)
            _ = np.load(os.path.join(outdir, "train_samples_y.npy"))
            if len(_)>1:
                train_y.append(_)
            _ = np.loadtxt(os.path.join(outdir, "val_samples_x.txt"))
            if len(_)>1:
                val_x.append(_)
            _ = np.load(os.path.join(outdir, "val_samples_y.npy"))
            if len(_)>1:
                val_y.append(_)
        
        try: 
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            val_x = np.concatenate(val_x)
            val_y = np.concatenate(val_y)
            train_y_last = np.concatenate([np.load(outdir_list[0]+"train_samples_y.npy")])
            if len(train_y_last)==0:
                train_y_last= train_y
        except:
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            val_x = np.array(val_x)
            val_y = np.array(val_y)
            train_y_last= train_y

        if usebest:
            trainbest_x= []
            trainbest_y=[]
            for outdir in outdir_list:
                _ = np.loadtxt(outdir+"best_samples_x.txt")
                if len(_)>1:
                    trainbest_x.append(_)
                _ = np.load(outdir+"best_samples_y.npy")
                if len(_)>1:
                    trainbest_y.append(_)
            try:
                trainbest_x = np.concatenate(trainbest_x)
                trainbest_y = np.concatenate(trainbest_y)
            except:
                trainbest_x = np.array(trainbest_x)
                trainbest_y = np.array(trainbest_y)

            if len(trainbest_x.shape)>1:
                if len(train_x.shape)>1:
                    train_x = np.concatenate([trainbest_x, train_x])
                    train_y = np.concatenate([trainbest_y, train_y])
                else:
                    train_x = trainbest_x
                    train_y = trainbest_y
                    train_y_last = trainbest_y
            valbest_x = np.concatenate([np.loadtxt(outdir+"best_samples_x_val.txt") for outdir in outdir_list])
            valbest_y = np.concatenate([np.load(outdir+"best_samples_y_val.npy") for outdir in outdir_list])
            if len(valbest_x.shape)>1:
                if len(val_x.shape)>1:
                    val_x = np.concatenate([valbest_x, val_x])
                    val_y = np.concatenate([valbest_y, val_y])
                else:
                    val_x = valbest_x
                    val_y = valbest_y
        print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
        if ypositive:
            train_y[np.where(train_y>1E10)] = 1E10
            train_y[np.where(train_y<1E-30)] = 1E-30
            train_y_last[np.where(train_y_last<1E-30)] = 1E-30
            val_y[np.where(val_y>1E10)] = 1E10
            val_y[np.where(val_y<1E-30)] = 1E-30
            bad_y= np.where(np.mean(train_y, axis=1)==1E-30)[0]
            bad_y_last= np.where(np.mean(train_y_last, axis=1)==1E-30)[0]
            if len(bad_y)>0:
                for item in bad_y:
                    train_y = np.delete(train_y,item,0)
                    train_x = np.delete(train_x,item,0)
            if len(bad_y_last)>0:
                for item in bad_y_last:
                    train_y_last = np.delete(train_y_last,item,0)

            bad_y= np.where(np.mean(val_y, axis=1)==1E-30)[0]
            if len(bad_y)>0:
                for item in bad_y:
                    val_y = np.delete(val_y,item,0)
                    val_x = np.delete(val_x,item,0)
        
        else:
            train_y[np.where(train_y>1E10)] = 1E10
            train_y[np.where(train_y<-1E5)] = -1E5
            val_y[np.where(val_y>1E8)] = 1E8
            val_y[np.where(val_y<-1E5)] = -1E5
            train_y_last[np.where(train_y_last>1E10)] = 1E10
            train_y_last[np.where(train_y_last<-1E5)] = -1E5
        
        X_mean = torch.tensor(XT1(train_x).mean(axis=0), dtype=torch.float32).to(device)
        X_std = torch.tensor(XT1(train_x).std(axis=0), dtype=torch.float32).to(device)
        X_transform  = X_transform_class(X_mean, X_std, device, dolog10index)
        X_transform.pickle(os.path.join(outdir_in,'X_transform.pkl'))
        if ypositive:
            y_mean = torch.tensor(torch.log(y_transform_data(torch.tensor(train_y,dtype=torch.float32).to(device))).median(axis=0).values, dtype=torch.float32).to(device)
            train_y_last[np.where(train_y_last==1E-30)]=np.median(train_y, axis=0)[np.where(train_y_last==1E-30)[1]]#
            y_std = torch.tensor(median_absolute_deviation(torch.log(y_transform_data(torch.tensor(train_y,dtype=torch.float32).to(device))), y_mean, 0), dtype=torch.float32).to(device)
        else:
            y_mean = torch.tensor(y_transform_data(torch.tensor(train_y_last,dtype=torch.float32).to(device)).median(axis=0).values, dtype=torch.float32).to(device)
            y_std = torch.tensor(median_absolute_deviation(y_transform_data(torch.tensor(train_y_last,dtype=torch.float32).to(device)), y_mean, 0), dtype=torch.float32).to(device)
        y_transform = Y_transform_class(y_mean, y_std, device, ypositive=ypositive)
        y_transform.pickle(os.path.join(outdir_in,'y_transform.pkl'))
        y_inv_transform = Y_invtransform_class(y_mean, y_std, data_tensor, device, ypositive=ypositive)
        y_inv_transform.pickle(os.path.join(outdir_in,'y_invtransform.pkl'))



        loss_fn = Loss_fn(data_tensor, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)
        val_metric_fn = Val_metric_fn(data_tensor, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)


        #Add user defined model
        linearmodel = None#LinearModel(norder, None, X_transform, y_transform, y_invtransform_data)
        train_xt = torch.from_numpy(train_x.astype(np.float32)).to(device)
        train_yt = torch.from_numpy(train_y.astype(np.float32)).to(device) 
        
        nnmodel = nnmodel_in(len(train_x[0]), len(train_y[0]), linearmodel, docpu=not docuda)
        if docuda:
            nnmodel = nnmodel.cuda()
        nnsampler.model=nnmodel
        model = train_nn(outdir_in, nnsampler.model, train_x, train_y, val_x, val_y, X_transform, y_transform, loss_fn, val_metric_fn, dev=device, verbose=True, retrain=retrain, pool=pool, nocpu=docuda, size=tsize, params=params)

def run_mcmc(nnsampler, outdir, method, ndim, nwalkers, init, log_prob, dlnp=None, ddlnp=None, pool=None, transform=None, ntimes=50, tautol=0.01, meanshift=0.1, stdshift=0.1, nk=2):
    """Run mcmc using the trained model 
    
    Args:
        nnsampler (``NN_samplerv1`` instance)
        outdir (string): output directory 
        method (string): mcmc method, only emcee or zeus is supported
        ndim (int): dimension of input parameters 
        nwalkers(int): number of walkers 
        init (array): numpy array with shape (nwalker, ndim) 
        log_probi (callable): likelihood
        pool (None or chtoPool): mpi pool
        transform (Transform or None): function to transform input
        ntimes (int): number of autocorrelation time
        tautol (float): tolerance of autocorrelation time error
        meanshift (float): maximum shifts of parameter mean estimations between first half and second half of the chains in unit of sigma
        stdshift (float): maximum shift of parameter error estimation between first half and second half of the chains in unit of percent
    """
    samp_steps = 5
    samp_eps = 0.1
    if method == "hmc":
        nnsampler._HMC_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, samp_steps, samp_eps, transform=transform)
    elif method  == "nuts":
        nnsampler._NUTS_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, 100, transform=transform)
    elif method == "emcee":
        nnsampler.emcee_sample(log_prob, ndim, nwalkers, init, pool, ntimes=ntimes, tautol=tautol, transform=transform, dlnp=dlnp, ddlnp=ddlnp, meanshift=meanshift, stdshift=stdshift, nk=nk)

    elif method == "zeus":
        nnsampler.Zeus_sample(log_prob, ndim, nwalkers, init, pool, ntimes=ntimes, tautol=tautol, transform=transform, dlnp=dlnp, ddlnp=ddlnp, meanshift=meanshift, stdshift=stdshift, nk=nk)
    else:
        nnsampler._HMC_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, samp_steps, samp_eps, transform=transform)

def logp_theory_data(samples, theory, data, invcov, logprior):
    """
        Internal function
    """
    logpall = []
    for t, s in zip(theory, samples):
        d = t-data 
        chisq = d.dot(invcov.dot(d))
        logpall.append(-0.5*chisq + logprior(s))
    return logpall


    






