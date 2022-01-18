import numpy as np
import pyDOE2
import sample_generator as sg
from copy import deepcopy
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
from schwimmbad import MPIPool
import glob
import pickle
import predictor_gpu
import sampler
import sys
import emcee
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from nn import *
from scipy.special import erf
from scipy.stats import chi2
import io
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import mkldnn as mkldnn_utils

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
def read_chain_and_cut(chainname, nk, ntimes=20, walkercut=False):
    reader = emcee.backends.HDFBackend(chainname, read_only=True)
    if nk > ntimes:
        print("Error: keep number greater then chain samples. nk: {0}, ntimes: {1}. This will lead to inclusion of all burn in step".format(nk, ntimes))
    nk = int(np.max(reader.get_autocorr_time(quiet=True))*nk)
    chain = reader.get_value("chain_transformed", discard=0, flat=False, thin=1)
    log_prob_samples = reader.get_log_prob(discard=0, flat=False, thin=1)
    if walkercut:
        good_walker_list = get_good_walker_list(log_prob_samples)
    else:
        good_walker_list = np.arange(0, len(log_prob_samples[0]))
    chain = chain[int(-1*nk):,good_walker_list,:].reshape(-1, len(chain[0,0]))
    log_prob_samples = log_prob_samples[int(-1*nk):, good_walker_list]
    return chain, log_prob_samples, reader

def _dummy_callback(x):
    pass 
class chtoPool(MPIPool):
    def __init__(self, comm=None):
        super(chtoPool, self).__init__(comm, use_dill=False)
        self.noduplicate=False
        self.worker_already_get_func_list=[]
    def wait(self):
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
                    print("reset", flush=True)
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

        Parameters
        ----------
        worker : callable
            A function or callable object that is executed on each element of
            the specified ``tasks`` iterable. This object must be picklable
            (i.e. it can't be a function scoped within a function or a
            ``lambda`` function). This should accept a single positional
            argument and return a single object.
        tasks : iterable
            A list or iterable of tasks. Each task can be itself an iterable
            (e.g., tuple) of values or data to pass in to the worker function.
        callback : callable, optional
            An optional callback function (or callable) that is called with the
            result from each worker run and is executed on the master process.
            This is useful for, e.g., saving results to a file, since the
            callback is only called on the master thread.

        Returns
        -------
        results : list
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
        self.worker_already_get_func_list=[]
        self.noduplicate = False
        workerset = self.workers.copy()
        for tid, workers in enumerate(workerset):
            self.comm.send(["reset", ["reset", None]], dest=workers, tag=tid)
    def bcast(self, worker, args, sizemax):
        workerset = self.workers.copy()
        for tid, workers in enumerate(workerset):
            if workers < sizemax:
                self.comm.send(["bcast", [worker, args]], dest=workers, tag=tid)
        worker(0)

##

def gauss2unif(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
def invgauss2unif(x):
    return np.sqrt(2)*torch.erfinv(2*x-1)

class Transform:
    def __init__(self, priors):
        self.priors = priors
    def __call__(self, x, returnnumpy=True, inputnumpy=True):
        """
        Transform perameters so that all the prior is gaussian with zero mean and unit variance 
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
    def __init__(self, priors):
        self.priors = priors
    def __call__(self, x, returnnumpy=True, inputnumpy=True):
        """
        Transform perameters so that all the prior is gaussian with zero mean and unit variance 
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
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i,:], self.y[i,:]

class Y_transform_data:
    def __init__(self, sigma, device):
        self.device= device
        self.sigma = torch.from_numpy(sigma.astype(np.float32)).to(device).clone().requires_grad_()
    
    def __call__(self, y):
        return y/self.sigma[None,:]
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)
    def transform_cov(self, cov):
        return torch.diag(1/self.sigma.type(torch.float64)).inner(cov).inner(torch.diag(1/self.sigma.type(torch.float64)))

class Y_invtransform_data:
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
    def __init__(self, X_mean, X_std, device, dolog10index=None):
        
        self.X_mean = X_mean
        self.X_std = X_std
        self.dev = device
        self.dolog10index = dolog10index 
        
    def __call__(self, X):
        #X= torch.tensor(X,dtype=torch.float32)
        X1 = X.clone()
        if self.dolog10index is not None:
            for ind in self.dolog10index:
                if len(X1.shape)>1:
                    X1[:,ind] = torch.log10(X[:,ind])
                else:
                    X1[ind] = torch.log10(X1[ind])
        return (X1 - self.X_mean[None,:].to(self.dev)) / self.X_std[None,:].to(self.dev)
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)
                        
class Y_transform_class:
    def __init__(self, y_mean, y_std, dev, ypositive=False):
        self.y_mean = y_mean
        self.y_std = y_std
        self.dev = dev
        self.ypositive = ypositive
        
    def __call__(self, y):
        if self.ypositive:
            return torch.exp(y * self.y_std[None,:].to(self.dev) + self.y_mean[None,:].to(self.dev))
        else:
            return y * self.y_std[None,:].to(self.dev) + self.y_mean[None,:].to(self.dev)
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            new = deepcopy(self)
            new.dev='cpu'
            pickle.dump(new,f , pickle.HIGHEST_PROTOCOL)

class Y_invtransform_class:
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
        if self.ypositive:
           expected = self.data_tensor 
           cov0 = (torch.diag(1/expected.type(torch.float64)).inner(cov).inner(torch.diag(1/expected.type(torch.float64)))).to(self.dev)
           cov0[cov0<=-1] = 1E-10-1
           cov1 = torch.log(1+cov0)
           #cov1 = cov0
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

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
def retrieve_model(outdir, inshape, outshape, nnmodel_in):
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
    






