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
from util import *

class NN_samplerv1:
    """

    """
    def __init__(self, model, outdir, prior_range):
        self.model = model
        self.outdir = outdir 
        self.prior_range = prior_range
        self.seed=123456
    def generate_training_data(self, samples, model, pool=None, args=None, kwargs=None):
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
        samples=[]
        Nsample_in = Nsamples
        shiftAs=False
        while(len(samples)<Nsamples):
            samples = pyDOE2.lhs(len(self.prior_range), samples=int(Nsample_in), criterion="center",
                                  iterations=5)
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
    def emcee_sample(self, log_prob, ndim, nwalkers, init, pool, transform, ntimes=50, tautol=0.01, dlnp=None, ddlnp=None):
        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform)
        if (ddlnp is not None)&(dlnp is not None):
            samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, 0, 0, outdir=self.outdir, overwrite=False, ntimes=ntimes, method = "emcee", incremental=True, progress=False, tautol=tautol)
    def HMC_sample(self, log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, transform, samp_steps, samp_eps):
        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform)
        samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, samp_steps, samp_eps, outdir=self.outdir, overwrite=True, ntimes=50, method = "hmc", incremental=True, progress=True)
    def NUTS_sample(self, log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, transform,  Madapt):
        max_n = 1000000
        x0 = init+0.1*np.random.randn(nwalkers, ndim)
        samp = sampler.HMCSampler(log_prob, dlnp, ddlnp, ndim, nwalkers, x0=x0, m=None, transform=transform, torchspeed=True)
        samp.calc_hess_mass_mat(maxiter=1E7, gtol=1E0)
        samp.sample(pool, max_n, 0,0,Madapt, outdir=self.outdir, overwrite=False, ntimes=50, method="nuts", incremental=True, progress=True)

class Log_prob:
    def __init__(self, data_new, invcov_new, model, y_invtransform_data, transform, temperature, nograd=False):
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
        self.noduplicate=True
    def __call__(self, x, returntorch=True, inputnumpy=True, returngrad=True):
        if not torch.is_tensor(x):
            x=torch.from_numpy(x.astype(np.float32)).to("cpu").clone().requires_grad_()
      #  x_in = torch.zeros(10)
      #  x_in[0:6]= torch.tensor([0.305, 2.27E-9, 0.98, 0.0505, 1E-3, 0.733],requires_grad=True)
        x_in  = self.transform(x,inputnumpy=False, returnnumpy=False)
        m = self.y_invtransform_data(self.model.predict(x_in,no_grad=self.no_grad))
        d = m-self.data_new

        like = (d@(self.invcov_new)@d.T*(-0.5))/self.T+lnprior(x)
        if torch.isnan(like[0][0]): 
            return -torch.inf
        else:
            if returntorch:
                return like[0][0]
            else:
                return like[0][0].detach().numpy() 
   
class Dlnp:
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
            loss = chisqMnn/chisqMd
            return loss, chisqMd, chisqnnd

class Loss_fn:
    def __init__(self, data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device):
        self.auxileryfunction = Auxilleryfunc(data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)
    def __call__(self, y_pred, y_target):
            loss, _, _ = self.auxileryfunction(y_pred, y_target)
            loss = torch.mean(loss)
            return loss

class Val_metric_fn:
    def __init__(self, data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device):
        self.auxileryfunction = Auxilleryfunc(data_in, cov_tensor, inv_cov_tensor, y_transform_data, y_inv_transform, device)
    def __call__(self, y_pred,y_target):
        loss, chisqMd, chisqnnd = self.auxileryfunction(y_pred,y_target)
        fracerr = torch.abs((chisqnnd)/chisqMd-1)
        return torch.tensor([torch.median(loss),torch.max(fracerr) ,torch.median(fracerr)])

class LogPrior:
    def __init__(self, prior):
        self.prior = prior 
    def __call__(self, xlist):
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
    return -0.5 * torch.sum(x.square())

def generate_training_point(theory, nnsampler, pool, outdir, ntrain, nval, chain=None, nsigma=1, omegab2cut=None, options=0):
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
                    samples = chain[np.random.randint(0, len(chain), ntrain)]
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
                    samples = chain[np.random.randint(0, len(chain), nval)]
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

def train_nn(outdir, model, train_x, train_y, val_x, val_y, X_transform, y_transform, loss_fn, val_metric_fn,dev = "cpu", verbose=False, retrain=True, pool=None, nocpu=False, size=0, rank=0, params=None):
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
    df = torch.abs(y-median)
    return df.median(axis=dim).values

def train_NN( nnsampler, cov, inv_cov, sigma, outdir_in, outdir_list,data, dolog10index=None, ypositive=False, retrain=True, norder=2, temperature=None, docuda=False, pool=None, tsize=1, nnmodel_in=None, params=None):
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




        train_x = np.concatenate([np.loadtxt(os.path.join(outdir, "train_samples_x.txt")) for outdir in outdir_list])
        train_y = np.concatenate([np.load(outdir+"train_samples_y.npy") for outdir in outdir_list])
        train_y_last = np.concatenate([np.load(outdir_list[0]+"train_samples_y.npy")])
        val_x = np.concatenate([np.loadtxt(os.path.join(outdir, "val_samples_x.txt")) for outdir in outdir_list])
        val_y = np.concatenate([np.load(outdir+"val_samples_y.npy") for outdir in outdir_list])
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


def run_mcmc(nnsampler, outdir, method, ndim, nwalkers, init, log_prob, dlnp=None, ddlnp=None, pool=None, transform=None, ntimes=50, tautol=0.01):
    #Run mcmc using the trained model 
    samp_steps = 5
    samp_eps = 0.1
    if method == "hmc":
        nnsampler.HMC_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, samp_steps, samp_eps, transform=transform)
    elif method  == "nuts":
        nnsampler.NUTS_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, 100, transform=transform)
    elif method == "emcee":
        nnsampler.emcee_sample(log_prob, ndim, nwalkers, init, pool, ntimes=ntimes, tautol=tautol, transform=transform, dlnp=dlnp, ddlnp=ddlnp)
    else:
        nnsampler.HMC_sample(log_prob, dlnp, ddlnp, ndim, nwalkers, init, pool, samp_steps, samp_eps, transform=transform)



def logp_theory_data(samples, theory, data, invcov, logprior):
    logpall = []
    for t, s in zip(theory, samples):
        d = t-data 
        chisq = d.dot(invcov.dot(d))
        logpall.append(-0.5*chisq + logprior(s))
    return logpall


def ml_sampler(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, outdir, theory, priors, data, cov,  init, pool, nwalkers, device, dolog10index, ypositive, temperatureArr, omegab2cut=None, docuda=False, tsize=1, gpunode=None, nnmodel_in=None, params=None):
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
    for i, (nt, nv, nk, ntimes, tautol, temperature) in enumerate(zip(ntrainArr, nvalArr, nkeepArr, ntimesArr, ntautolArr, temperatureArr)):
        temperature = temperature**2
        print("#"*100)
        print("iteration: {0}".format(i), flush=True)
        print("#"*100)
        outdir_in = os.path.join(outdir, "iter_{0}/".format(i))
        if i==0:
            chain=None
        else:
            chain_name = os.path.join(os.path.join(outdir, "iter_{0}/".format(i-1)), "chemcee_256")
            if os.path.isfile(chain_name+".h5"):
                chain_name = chain_name+".h5"
                chain, _temp, _temp2= read_chain_and_cut(chain_name.format(i-1), nk, ntimes)
            else:
                chain_name = chain_name+".txt"
                chain = np.loadtxt(chain_name)[-100000:,:-1]
        #Generate training 
        ntrain = nt
        nval = nv
        nnsampler = NN_samplerv1(None,  outdir_in, prior_range) 
        if "trainingoption" in params:
            options = params['trainingoption']
        else:
            options = 0
        generate_training_point(theory, nnsampler, pool, outdir_in, ntrain, nval, chain, nsigma=3, omegab2cut=omegab2cut, options=options)
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
            pickle.dump([nnsampler, cov, inv_cov, sigma, outdir_in, outdir_list, data, dolog10index, ypositive, False, 2, temperature, True, None, 1, nnmodel_in, params], f) 
            f.close()
            if not os.path.isfile(outdir_list[-1] + "/finish.pkl"): 
                print("running gpu on {0}".format(gpunode), flush=True)
                os.system("cat /home/users/chto/code/lighthouse/python/nnacc/nnacc/train_gpu.py | ssh {0} python - {1}".format(gpunode, outdir_list[-1]))
                while(1):
                    if  os.path.isfile(outdir_list[-1] + "/finish.pkl"):
                        break


        
        #Retrieve model  
        model, y_invtransform_data = retrieve_model(outdir_in, len(init), len(data), nnmodel_in)
        if not docuda:
            model.model = model.model.to(memory_format=torch.channels_last)
            model.MKLDNN=True


        #Do MCMC
        
        if os.path.isfile(outdir_in+"/chemcee_256.h5"):
            continue

        invcov_new = torch.from_numpy(inv_cov.astype(np.float32)).to('cpu').detach().clone().requires_grad_()
        data_new = torch.from_numpy(data.astype(np.float32)).to('cpu').detach().clone().requires_grad_()


        log_prob = Log_prob(data_new, invcov_new, model, y_invtransform_data, transform, temperature, nograd=True) 
        dlnp = Dlnp(data_new, invcov_new, model, y_invtransform_data, transform, temperature)
        ddlnp = Ddlnp(data_new, invcov_new, model, y_invtransform_data, transform, temperature)
        dlnp = None
        ddlnp = None
        if pool is not None:
            pool.noduplicate=True
        run_mcmc(nnsampler, outdir_in, sys.argv[1], ndim, nwalkers, init, log_prob, dlnp=dlnp, ddlnp=ddlnp, pool=pool, transform=transform, ntimes=ntimes, tautol=tautol)
        if pool is not None:
            pool.noduplicate_close() 
    if 'nimp' in params.keys():
        if not os.path.isfile(outdir+"/samples_im.npy"):
            chain_name = os.path.join(os.path.join(outdir, "iter_{0}/".format(len(ntrainArr)-1)), "chemcee_256")
            if os.path.isfile(chain_name+".h5"):
                chain_name = chain_name+".h5"
                chain, log_prob_samples_x, reader = read_chain_and_cut(chain_name.format(len(ntrainArr)-1), nk, ntimes)
                log_prob_samples_x = reader.get_log_prob(discard=0, flat=True, thin=1)
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
        nnsampler = NN_samplerv1(None,  outimp, prior_range)
        if not os.path.isdir(outimp):
            os.makedirs(outimp)
        if not os.path.isfile(outdir+"/theory.npy"):
            theory = nnsampler.generate_training_data(zip(range(len(chain)), chain), theory, pool=pool, args=[outimp])
            np.save(outdir+"/theory.npy", theory)
        else:
            theory = np.load(outdir+"/theory.npy")
        logprior=LogPrior(priors)
        logp = logp_theory_data(chain, theory, data, inv_cov, logprior)
        w = np.exp(logp-log_prob_samples_x)
        w[np.abs(np.log(w)-np.mean(np.log(w)))>2*np.std(np.log(w))]=0
        w = w/np.sum(w)
        np.save(outdir+"/weight_im.npy", [log_prob_samples_x, logp, w])





