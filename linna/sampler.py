import numpy as np
from tqdm import tqdm
from emcee.state import State
import emcee
import os
from scipy.optimize import minimize
from functools import wraps
import torch
import zeus
from zeus.autocorr import AutoCorrTime
import copy
import h5py

def stop_criterion(thetaminus, thetaplus, rminus, rplus, cov=None):
    raise NotImplementedError()

def leapfrog(theta, r, grad, epsilon, f, cov=None):
    raise NotImplementedError()

def build_tree(theta, r, grad, v, j, epsilon, f, joint0, cov):
    raise NotImplementedError()

class Functransform:
    def __init__(self, xmap, u, func, derive=0, torchspeed=False):
        self.u = u
        self.xmap = xmap
        self.orig_lnp = func
        self.derive = derive
        self.torchspeed = torchspeed
    def __call__(self, x, *args, **kwargs):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x.astype(np.float32)).to('cpu').clone().requires_grad_()
        xp = torch.from_numpy(self.xmap.astype(np.float32)).to('cpu').clone().requires_grad_()+torch.from_numpy(self.u.astype(np.float32)).to('cpu').clone().requires_grad_() @ x
        
        if self.derive==1:
            if torch.is_tensor(x):
                xp = xp.detach().numpy()
            return self.u.T @ (self.orig_lnp( xp, *args, **kwargs))
        elif self.derive==2:
            if torch.is_tensor(x):
                xp = xp.detach().numpy()
            return self.u.T @ (self.orig_lnp( xp, *args, **kwargs)) @ self.u
        else:
            if not self.torchspeed:
                lnp = self.orig_lnp( xp.detach().numpy(), *args, **kwargs)
                return lnp.item()
            else:
                lnp = self.orig_lnp( xp, *args, **kwargs)
                return lnp.item(), torch.autograd.grad(lnp,x)[0].detach().numpy()

class CombineFunc:
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
    def __call__(self, x, *args, **kwargs):
        return self.f1(x, *args, **kwargs), self.f2(x, *args, **kwargs)


class _hmc_wrapper(object):
    def __init__(self, random, model_derive, cov, epsilon, nsteps=None):
        self.random = random
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.cov = _hmc_matrix(np.asarray(cov))
        self.model_derive = model_derive #grad_log_probability

    def __call__(self, args):
        coords, current_p = args

        # Sample the initial momentum.
        current_q = coords
        q = current_q
        p = current_p

        # Initial leapfrog step.
        grad_log_probability = self.model_derive(coords)
        p = p + 0.5 * self.epsilon * grad_log_probability

        # Continue leapfrog steps.
        for i in range(self.nsteps):
            # First, a full step in position.
            q = q + self.epsilon * self.cov.apply(p)

            # Then a full step in momentum.
            if i < self.nsteps - 1:
                grad_log_probability = self.model_derive(q)
                p = p + self.epsilon * grad_log_probability

        # Finish with a half momentum step to synchronize with the position.
        grad_log_probability = self.model_derive(q)
        p = p + 0.5 * self.epsilon * grad_log_probability


        # Compute the acceptance probability factor.
        #Factor is the kinetic energy term 
        factor = 0.5 * np.dot(current_p, self.cov.apply(current_p))
        factor -= 0.5 * np.dot(p, self.cov.apply(p))
        return q,  factor


class HamiltonianMove(emcee.moves.Move):
    def __init__(self, compute_derivative, nsteps, epsilon, cov):
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.cov = cov
        self.compute_derivative = compute_derivative

    def get_args(self, ensemble):
        # Randomize the stepsize if requested.
        rand = ensemble.random
        try:
            eps = float(self.epsilon)
        except TypeError:
            eps = rand.uniform(self.epsilon[0], self.epsilon[1])

        # Randomize the number of steps.
        try:
            L = int(self.nsteps)
        except TypeError:
            L = rand.randint(self.nsteps[0], self.nsteps[1])

        return eps, L

    def propose(self, model, state):
        # Set up the integrator and sample the initial momenta.
        nwalkers, ndim = state.coords.shape
        integrator = _hmc_wrapper(model.random, self.compute_derivative,self.cov, *(self.get_args(model)))
        momenta = integrator.cov.sample(model.random, nwalkers,
                                        ndim)

        # Integrate the dynamics in parallel.
        newq = np.zeros((nwalkers, ndim))
        newfactor = np.zeros(nwalkers)
        #for i, (coord, mome) in enumerate(zip(state.coords, momenta)):
        test  = list(model.map_fn(integrator, zip(state.coords, momenta)))
        for n in range(nwalkers):
            newq[n], newfactor[n] = test[n]
        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(newq)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob +  newfactor
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff
        
        # Update the parameters
        new_state = State(newq, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted

def find_reasonable_epsilon(theta0, grad0, logp0, f, cov):
    """ 
        Heuristic for choosing an initial value of epsilon 
    """
    epsilon = 1.
    r0 = np.random.normal(0., 1., len(theta0))

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f, cov)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    k = 1.
    while np.isinf(logpprime) or np.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f, cov)

    epsilon = 0.5 * k * epsilon

    # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
    # a = 2. * float((acceptprob > 0.5)) - 1.
    logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, cov.apply(rprime))-np.dot(r0,cov.apply(r0)))
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f, cov)
        # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, cov.apply(rprime))-np.dot(r0,cov.apply(r0)))

    print("find_reasonable_epsilon=", epsilon)

    return epsilon

class NUTSMove(emcee.moves.Move):
    #Modified from  Mfouesneau 's awesome Nuts package 
    #https://github.com/mfouesneau/NUTS/blob/master/nuts/nuts.py
    def __init__(self, lnp, compute_derivative, cov, Madapt, x0, nwalkers, delta=0.6, maxheight=np.inf, torchspeed=True):
        self.cov = _hmc_matrix(np.asarray(cov))
        self.lnp = lnp
        self.dlnp = compute_derivative
        f = self.lnp
        self.epsilon = np.zeros(nwalkers)
        for n, x in enumerate(x0):
            logp, grad = f(x)
            self.epsilon[n] = find_reasonable_epsilon(x, grad, logp, f, self.cov)
        # Parameters to the dual averaging algorithm.
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75
        self.mu = np.log(10. * self.epsilon)

        # Initialize dual averaging algorithm.
        self.epsilonbar = np.ones(nwalkers)
        self.Hbar = np.zeros(nwalkers)
        self.Madapt = Madapt
        self.m = np.ones(nwalkers)
        self.delta = delta
        self.maxheight = maxheight
        self.torchspeed=torchspeed


    def propose(self, model, state):
        # Set up the integrator and sample the initial momenta.
        nwalkers, ndim = state.coords.shape
        r0s = self.cov.sample(model.random, nwalkers,ndim)

        coords = state.coords
        returncoords=np.zeros_like(state.coords) 
        returnlogps = np.zeros_like(state.log_prob) 
        accepteds =  np.zeros_like(state.log_prob)>1
        returngrad= np.zeros_like(state.blobs)
        function_in = Nuts_one_worker(state, r0s, state.coords, self.dlnp, self.epsilon, model.random, self.cov,  self.maxheight, model.log_prob_fn, self.torchspeed)
    
        test = list(model.map_fn(function_in, range(nwalkers)))
        for n in range(nwalkers):
            returncoords[n], returnlogps[n], accepteds[n], alpha, nalpha, returngrad[n] = test[n]
            if (self.m[n] <= self.Madapt):
                eta = 1. / float(self.m[n] + self.t0)
                self.Hbar[n] = (1. - eta) * self.Hbar[n] + eta * (self.delta - alpha / float(nalpha))
                self.epsilon[n] = np.exp(self.mu[n] - np.sqrt(self.m[n]) / self.gamma * self.Hbar[n])
                eta = self.m[n] ** -self.kappa
                self.epsilonbar[n] = np.exp((1. - eta) * np.log(self.epsilonbar[n]) + eta * np.log(self.epsilon[n]))
            elif self.m[n] == self.Madapt+1:
                print("final epsilon={0} for walker:{1} ".format(self.epsilonbar[n], n))
                self.epsilon[n] = self.epsilonbar[n]
            else:
                pass
            self.m[n] += 1


        new_state = State(returncoords, log_prob=returnlogps, blobs=returngrad)
        state = self.update(state, new_state, accepteds)

        return state, accepteds

class Nuts_one_worker(object):
    def __init__(self, state, r0s, coords, dlnp, epsilon, random, cov,  maxheight, log_prob_fn, torchspeed):
        self.state = state
        self.r0s = r0s
        self.coords = coords
        self.dlnp = dlnp
        self.random = random
        self.cov = cov

        # Initialize dual averaging algorithm.
        self.epsilon = epsilon 
        self. maxheight =  maxheight
        self.log_prob_fn = log_prob_fn
        self.torchspeed = torchspeed

    def f(self, x):
            #x=torch.from_numpy(x.astype(np.float32)).to('cpu').clone().requires_grad_()
            lnP, grad = self.log_prob_fn(x)
            #grad = torch.autograd.grad(lnP, x)[0]
            return lnP, grad 
    def __call__(self, n):
            accepted=False
            joint = self.state.log_prob[n] - 0.5 * np.dot(self.r0s[n].T, self.cov.apply(self.r0s[n]))
            coord = self.coords[n]
            grad = self.dlnp(coord)#self.state.blobs[n] 
            r0 = self.r0s[n]
            # initialize the tree
            thetaminus = coord[:]
            thetaplus = coord[:]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            s = 1  # Main loop: will keep going until s == 0.
            returncoord = coord
            returnlogp = self.state.log_prob[n]
            returngrad = self.state.blobs[n]
            epsilon= self.epsilon[n]
            logptree =0
            while (s==1 and j<self.maxheight):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (self.random.uniform() < 0.5) - 1)
                if (v == -1):  thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(thetaminus, rminus, gradminus, v, j, epsilon, self.f, joint, self.cov)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(thetaplus, rplus, gradplus, v, j, epsilon, self.f, joint, self.cov)
                logptot = np.logaddexp(logptree, logptree2)
                if (sprime == 1) and (np.log(self.random.uniform()) < logptree2 - logptot):
                    accepted=True
                    returncoord = thetaprime[:]
                    returnlogp = logpprime
                    returngrad = gradprime[:]
                logptree = logptot
                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus, self.cov)
                # Increment depth.
                j += 1
            # Do adaptation of epsilon if we're still doing burn-in.
            return returncoord, returnlogp, accepted, alpha, nalpha, returngrad



class _hmc_matrix(object):

    def __init__(self, var):
        self.var = var

    def sample(self, random, *shape):
        return random.randn(*shape) * np.sqrt(self.var)

    def apply(self, x):
        return x/self.var

class Transformbackend(emcee.backends.HDFBackend):
    def __init__(self,
        filename,
        transform, name="mcmc", read_only=False, dtype=None
     ):
        super(Transformbackend, self).__init__(filename, name, read_only, dtype)
        self.transform = transform

    def reset(self,nwalkers, ndim):
        super().reset(nwalkers, ndim)
        with self.open("a") as f:
            g = f[self.name]
            g.create_dataset(
                "chain_transformed",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
            )
    def grow(self, ngrow, blobs):
        super().grow(ngrow, blobs)
        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain_transformed"].resize(ntot, axis=0)
    def save_step(self, state, accepted):
        """Save a step to the backend
        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
        """
        self._check(state, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            g["chain"][iteration, :, :] = state.coords #[self.transform(c) for c in state.coords]
            g["chain_transformed"][iteration, :, :] = [self.transform(c) for c in state.coords]
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1
        
def checkmeanstd(samples, meanshift, stdshift):
    """
    Check convergence of the mcmc samples by comparing mean and standard deviation estimations of the first half of the sample and the second half 
    
    Args:
       samples (ndarray): (nstep, nwalker, nparam) array. 
       meanshift (float): criteria on shifts in mean 
       stdshift ( float): criteria on shifts in standard deviation

    Returns:
        bool: converged or not 
    """
    firsthalf = samples[:int(len(samples)/2)].reshape(-1, samples.shape[-1])
    secondhalf = samples[int(len(samples)/2):].reshape(-1, samples.shape[-1])
    meanshifte = np.max(np.abs(np.mean(firsthalf, axis=0)-np.mean(secondhalf, axis=0))/np.std(secondhalf, axis=0))
    stdshifte = np.max((np.std(firsthalf, axis=0)-np.std(secondhalf, axis=0))/np.std(secondhalf, axis=0))
    return (meanshifte<meanshift)&(stdshifte<stdshift)

class HMCSampler:
    def __init__(self, lnp, dlnp, ddlnp, ndim, nwalkers,  x0=None, m=None, transform=None, torchspeed=False):
        self.lnp = lnp
        self.dlnp = dlnp
        self.ddlnp = ddlnp
        self.transform = transform
        self.x0 = x0
        self.nparams = ndim
        self.nwalkers=nwalkers
        self.sampler=None


        if m is None:
            self.m = np.ones(self.nparams)
        else:
            self.m = m
        self.torchspeed = torchspeed
        self.calc_hess=False

    def calc_hess_mass_mat(self, maxiter=1E5, gtol=1E-5,resamp_x0=True, tensor=True):
        x = self.x0[0]

        #pbar = tqdm(range(nsteps))
        #for i in pbar:
        #    lnp = self.lnp(x)
        #    grad = self.dlnp(x)
        #    x = x + grad * eps
        #    pbar.set_description('log-prob: {}'.format(lnp))

        res = minimize(lambda x: -1*self.lnp(x).detach().numpy(), x, method='Nelder-Mead',
               options={'maxiter':maxiter, 'disp': True, 'gtol':gtol})

        res = minimize(lambda x: -1*self.lnp(x).detach().numpy(), res.x, method='BFGS',
               jac=lambda x: -1*self.dlnp(x), 
               options={'maxiter':maxiter, 'disp': True, 'gtol':gtol})
        print("optimization done")
        print("#"*100)
        print("#"*100)

        hess = []
        lnp = self.lnp(x)
        hess = self.ddlnp(x)

        u, m, _ = np.linalg.svd(-hess)
        s = 1 / m

        self.u = u
        self.m = m
        self.orig_lnp = self.lnp
        self.orig_dlnp = self.dlnp
        self.orig_ddlnp = self.ddlnp
        self.xmap = x
        self.lnp_in = Functransform(self.xmap, self.u, self.orig_lnp, torchspeed=self.torchspeed)
        self.dlnp = Functransform(self.xmap, self.u, self.orig_dlnp, derive=1)
        if not self.torchspeed:
            self.lnp = CombineFunc(self.lnp_in, self.dlnp)
        else:
            self.lnp = self.lnp_in
        self.ddlnp = Functransform(self.xmap, self.u, self.orig_ddlnp, derive=2)
        if self.transform is None:
            self.transform = lambda x: self.xmap + self.u @ x
        else:
            self.orig_transform = self.transform
            self.transform = lambda x: self.orig_transform(self.xmap + self.u @ x)

        if resamp_x0:
            self.x0 = np.random.randn(*self.x0.shape) *0.5*np.sqrt(s)
        self.calc_hess=True

    def sample(self, pool, nsamp, samp_steps, samp_eps, Madapt=1000, outdir="./", progress=False, overwrite=False, ntimes=10, tautol=0.01, method="hmc", incremental=True, meanshift=0.1, stdshift=0.1, nk=2):
        if (not self.calc_hess)&(method=="nuts"):
           self.lnp = CombineFunc(self.lnp, self.dlnp) 
           self.calc_hess=True
        if self.transform is None:
            self.transform = lambda x: x
        x0 = self.x0
        
        if method=="hmc":
            filename = os.path.join(outdir, "chhmc.h5")
        elif method=="emcee":
            filename = os.path.join(outdir, "chemcee_256.h5")
        elif method == "nuts":
            filename = os.path.join(outdir, "chnuts_test6.h5")
        else:
            print("unknown method: ", method)
            sys.exit()
        if os.path.isfile(filename):
            if overwrite:
                if pool.is_master():
                    os.remove(filename)
            else:
                print("init from previous")
                x0=None
        backend = Transformbackend(filename, self.transform)
        if x0 is None:
            x0 = backend.get_last_sample()
            resume = True
        else:
            resume = False
        dtype = None#[("log_prior", float)]
        if not incremental:
            backend=None
        if method=="hmc":
            moves = [(HamiltonianMove(self.dlnp, samp_steps, samp_eps, self.m),1)] 
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.lnp, pool=pool,  backend=backend, blobs_dtype=dtype, moves=moves)
        elif method=="emcee":
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.lnp, pool=pool,  backend=backend, blobs_dtype=dtype)
        elif method == "nuts":
            if self.torchspeed:
                kwargs = {"returntorch":True}
            else:
                kwargs = {}
            print("########")
            moves = [(NUTSMove(self.lnp, self.dlnp,self.m, Madapt=Madapt, x0=x0, nwalkers=x0.shape[0], torchspeed=self.torchspeed, maxheight=5),1)] 
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.lnp, pool=pool,  backend=backend, blobs_dtype=dtype, moves=moves,kwargs=kwargs)
            
        else:
            print("this should never happen")
            
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(nsamp)

        # This will be useful to testing convergence
        old_tau = np.inf
        print("start",flush=True)
        if not incremental:
            self.sampler.run_mcmc(x0,nsteps=nsamp, progress=progress, skip_initial_state_check=True);
            return np.array([self.transform(c) for c in self.sampler.get_chain(flat=True)])
        else:
            if not resume:
                print("burnin...", flush=True)
                nwalker = x0.shape[0]
                sampler = emcee.EnsembleSampler(self.nwalkers, self.nparams, self.lnp, pool=pool, blobs_dtype=dtype)
                _ = sampler.run_mcmc(x0,nsteps=100, progress=True, skip_initial_state_check=True);
                flat_chain = sampler.get_chain(flat=True)
                log_prob = sampler.get_log_prob(flat=True)
                pos = flat_chain[np.argsort(log_prob)[::-1][:int(50*nwalker)]]
                x0 = pos[np.random.randint(0,len(pos),nwalker),:]
                print("burnin done...", flush=True)
                self.sampler.reset()    
            for sample in self.sampler.sample(x0, iterations=nsamp, progress=progress, skip_initial_state_check=True):
                # Only check convergence every 100 steps
                if self.sampler.iteration % 100:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                
                if np.isnan(np.sum(tau)) and (self.sampler.iteration>10):
                    break
                # Check convergence
                converged = np.all(tau * ntimes< self.sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < tautol)
                converged &= checkmeanstd(self.sampler.get_chain()[-int(nk*np.mean(tau)):], meanshift=meanshift, stdshift=stdshift)
                print("max, min tau diff, max tau, ninter: {0}, {1}, {2}, {3}\n".format(np.max(np.abs(old_tau - tau) / tau), np.min(np.abs(old_tau - tau) / tau), np.max(tau), self.sampler.iteration), flush=True)
                if converged:
                    break

                old_tau = tau
            del self.sampler
            self.sampler = None

class ZeusTransformCallback(zeus.callbacks.SaveProgressCallback):
    def __init__(self, filename, ncheck, transform):
        super(ZeusTransformCallback, self).__init__(filename, ncheck)
        self.transform = transform
    
    def __save(self, x, y):
        x = copy.deepcopy(x)
        with h5py.File(self.directory, 'a') as hf:
            hf['samples'].resize((hf['samples'].shape[0] + x.shape[0]), axis = 0)
            hf['samples'][-x.shape[0]:] = x
            hf['chain_transformed'].resize((hf['samples'].shape[0] + x.shape[0]), axis = 0)
            hf['chain_transformed'][-x.shape[0]:] = np.apply_along_axis(self.transform, -1, x)
            hf['logprob'].resize((hf['logprob'].shape[0] + y.shape[0]), axis = 0)
            hf['logprob'][-y.shape[0]:] = y


    def __initialize_and_save(self, x, y):
        x = copy.deepcopy(x)
        with h5py.File(self.directory, 'w') as hf:
            hf.create_dataset('samples', data=x, compression="gzip", chunks=True, maxshape=(None,)+x.shape[1:])
            hf.create_dataset('chain_transformed', data=np.apply_along_axis(self.transform, -1, x), compression="gzip", chunks=True, maxshape=(None,)+x.shape[1:])
            hf.create_dataset('logprob', data=y, compression="gzip", chunks=True, maxshape=(None,)+y.shape[1:]) 
        self.initialised  = True
    def __call__(self, i, x, y):
        """
        Method that calls the callback function.
        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        if i % self.ncheck == 0:
            if self.initialised:
                self.__save(x[i-self.ncheck:], y[i-self.ncheck:])
            else:
                self.__initialize_and_save(x[i-self.ncheck:], y[i-self.ncheck:])

        return None

    def get_last_sample(self):
        with h5py.File(self.directory, "r") as hf:
            samples = np.copy(hf['samples'])
        return samples[-1]

class Zeusbackend:
    def __init__(self, chainname):
        self.name = chainname

    def get_value(self, name, flat=False, thin=1, discard=0):
        with h5py.File(self.name, "r") as hf:
            v = np.copy(hf[name])
        v = v[discard + thin - 1 : len(v) : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        else:
            return v
    def get_log_prob(self, **kwargs):
        return self.get_value("logprob", **kwargs)

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        """
        Args:
            c (float, optional): Truncation parameter of automated windowing procedure of Sokal (1989), default is 5.0
            method (str):  Method to use to compute the integrated autocorrelation time. Available options are ``mk`` (Minas Karamanis method), ``dfm (Daniel Forman-Mackey method)``, and ``gw (Goodman-Weary method)``.

        """
        x = self.get_value("samples", discard=discard, thin=thin)        
        return thin * AutoCorrTime(x, **kwargs)


class ZeusconvergenceCallback:
    """
    Modified from Zeus's autoconvergence callback
    Args:
        ncheck (int): The number of steps after which the IAT is estimated and the tests are performed.
            Default is ``ncheck=100``.
        dact (float): Threshold of the rate of change of IAT. Sampling terminates once this threshold is
            reached along with the other criteria. Default is ``dact=0.01``.
        nact (float): Minimum lenght of the chain as a mutiple of the IAT. Sampling terminates once this threshold is
            reached along with the other criteria. Default is ``nact=10``.
        discard (float): Percentage of chain to discard prior to estimating the IAT. Default is ``discard=0.5``.
        trigger (bool): If ``True`` (default) then terminatate sampling once converged, else just monitor statistics.
        method (str): Method to use for the estimation of the IAT. Available options are ``mk`` (Default), ``dfm``, and ``gw``.
        samples (ndarray): (nstep, nwalker, nparam) array. 
        meanshift (float): criteria on shifts in mean 
        stdshift ( float): criteria on shifts in standard deviation
    """

    def __init__(self, ncheck=100, dact=0.01, nact=10, discard=0.5, trigger=True, method='mk', meanshift=0.1, stdshift=0.1, nk=2):
        self.ncheck = ncheck
        self.dact = dact 
        self.nact = nact

        self.discard = discard
        self.trigger = trigger
        self.method = method

        self.estimates = []
        self.old_tau = np.inf

        self.meanshift = meanshift
        self.stdshift = stdshift
        self.nk=nk

    def __call__(self, i, x, y):
        """
        Method that calls the callback function.
        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        converged = False

        if i % self.ncheck == 0:
        
            tau = np.mean(AutoCorrTime(x[int(i * self.discard):], method=self.method))
            self.estimates.append(tau)

            # Check convergence
            converged = tau * self.nact < i
            converged &= np.abs(self.old_tau - tau) / tau < self.dact
            converged &= checkmeanstd(x[-int(self.nk*tau):], self.meanshift, self.stdshift)
            self.old_tau = tau

        if self.trigger:
            return converged
        else:
            return None


class ZeusSampler:
    def __init__(self, lnp, ndim, nwalkers,  x0=None, transform=None):
        self.lnp = lnp
        self.transform = transform
        self.x0 = x0
        self.nparams = ndim
        self.nwalkers=nwalkers
        self.sampler=None

    def sample(self, pool, nsamp, outdir="./", progress=False, overwrite=False, ntimes=10, tautol=0.01, incremental=True, meanshift=0.1, stdshift=0.1, nk=2):
        if self.transform is None:
            self.transform = lambda x: x
        x0 = self.x0
        filename = os.path.join(outdir, "zeus_256.h5")
        if os.path.isfile(filename):
            if overwrite:
                if pool.is_master():
                    os.remove(filename)
            else:
                print("init from previous")
                x0=None
        backend = ZeusTransformCallback(filename, 100, self.transform)
        if x0 is None:
            x0 = backend.get_last_sample()
            resume = True
        else:
            resume = False
        if not incremental:
            backend=None
        self.sampler = zeus.EnsembleSampler(self.nwalkers, self.nparams, self.lnp, pool=pool, maxiter=1E5)#, moves=[zeus.moves.GlobalMove()])
        cb0 = ZeusconvergenceCallback(ncheck=100, dact=tautol, nact=ntimes, discard=0.2, method='dfm', meanshift=meanshift, stdshift=stdshift, nk=nk)

        if not incremental:
            self.sampler.run_mcmc(x0,nsteps=nsamp, progress=progress);
            return np.array([self.transform(c) for c in self.sampler.get_chain(flat=True)])
        else:
            self.sampler.run_mcmc(x0, nsamp, callbacks=[backend, cb0]);
            del self.sampler
            self.sampler = None
