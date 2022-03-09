import torch
from torch import nn
from linna.nn import *
from tqdm.auto import tqdm
from linna.nnutils import *
from torch_lr_finder import LRFinder
import copy
from torch.utils import mkldnn as mkldnn_utils
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import *
from numpy import diff




class EarlyStopping(object):
    """
    class handles conditions of terminating neural network training 
    """
    def __init__(self, mode='min', min_delta=0, patience=10, nqueue=200,  percentage=False):
        """
        Args:
            mode (string): Modes to determine whether the loss function is decreasing or not. 
            min_delta (float):  The loss is decreasing if the new score is less than best score - best*min_delta/100 (percentage == False) or less than best score - best*min_delta (percentage==True) 
            parience (int): if best score is not updated after patience steps, the nn will stop training
            nqueue (int): number of step to estimate derivative 
            percentage (bool): determine how min_delta is used 
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_t = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.cooling = 0
        self.cooling_weight_decay = 0
        self.queue_t = []
        self.queue_v = []
        self.nqueue = nqueue 

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, metrics_t):
        """
        Args:
             metrics (float): validation score 
             metrics_t (float): training score
        """
        self.queue_t.append(metrics_t)
        self.queue_v.append(metrics)
        if len(self.queue_t)>self.nqueue:
            self.queue_t.pop(0)
        if len(self.queue_v)>self.nqueue:
            self.queue_v.pop(0)
        running_avg_t = torch.median(torch.stack(self.queue_t))
        running_avg_v = np.median(self.queue_v)
        if len(self.queue_t)>2:
            firsthalft = torch.median(torch.stack(self.queue_t[:int(0.5*len(self.queue_t))]))
            secondhalft = torch.median(torch.stack(self.queue_t[int(0.5*len(self.queue_t)):]))
            firsthalfv = np.median(self.queue_v[:int(0.5*len(self.queue_v))])
            secondhalfv = np.median(self.queue_v[int(0.5*len(self.queue_v)):])
        if self.best is None:
            self.best = metrics
            self.best_t = metrics_t
            self.num_bad_epochs = 0 
        
            return 0
        if np.isnan(metrics):
            print("nan metric", flush=True)
            self.num_bad_epochs += 1
            return 0
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.cooling =0 
            self.cooling_weight_decay = 0
            self.best = metrics
            self.best_t = metrics_t
        else:
            self.num_bad_epochs += 1
            if (self.num_bad_epochs>=self.patience*0.9)&(self.num_bad_epochs<self.patience): 
#                if self.is_better(metrics_t, self.best_t):
#                    self.num_bad_epochs-=1
#                    if self.cooling_weight_decay!=0:
#                        if self.cooling_weight_decay>500:
#                            self.cooling_weight_decay=0
#                            return 0
#                        else:
#                            self.cooling_weight_decay+=1
#                            return 0
#                    else:
#                        self.cooling_weight_decay+=1    
#                        return 3
#                else:
                if (self.cooling!=0):
                    if self.cooling>500:
                        self.cooling=0
                        self.num_bad_epochs += 5
                        return 0
                    else:
                        self.num_bad_epochs-=1
                        self.cooling+=1
                        return 0
                    
                else:
                    self.cooling+=1
                    return 1
            if len(self.queue_t)>2 :
                if  (len(self.queue_t)>0.5*self.nqueue) and (secondhalft-firsthalft <0 ) and (secondhalfv-firsthalfv >0):
                        if self.cooling_weight_decay!=0:
                            if self.cooling_weight_decay>1000:
                                self.cooling_weight_decay=0
                                return 0
                            else:
                                self.queue_t = []
                                self.queue_v = []
                                self.cooling_weight_decay+=1

                                if self.cooling_weight_decay%50==0:
                                    return 3
                                else:
                                    return 0
                        else:
                            self.cooling_weight_decay+=1    
                            return 3
        if self.num_bad_epochs >= self.patience:
            return 2
        return 0

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class Predictor:
    """
    class handle neural network training and prediction
    """
    def __init__(self, in_size=None, out_size=None, model=None, optim=None, X_transform=None,
                 y_transform=None, device='cpu', scheduler=None, outdir=None):
        """
        Args:
            in_size (int): input size of the neural network 
            out_size (int): output size of the neural network 
            model (string): specify the neural network model defined in nn.py
            optim (pytorch optimizer instance or string, optional): default one is AdamW, if "automatic": will use AdamW with automatically determined learning rate 
            X_transform (callable): transformation function of the input, whose output will be fed into neural network 
            y_transform (callable): transformation function of the output, whouse ouput will be used to training neural network
            device (string): cpu or cuda
            scheduler (None): not used for now [reserved for mpi training]
            outdir (string): specify the output directory
        """
        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.best_val_loss= float('inf')
        self.outdir = outdir
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = ChtoModel(in_size, out_size).to(device)
                    
        self.scheduler=scheduler

        if optim is not None:
            
            self.optim = optim
        else:
            self.optim = torch.optim.AdamW(self.model.parameters())

        if X_transform is not None:
            self.X_transform = X_transform
        else:
            self.X_transform = lambda x: x

        if y_transform is not None:
            self.y_transform = y_transform
        else:
            self.y_transform = lambda x: x
        self.MKLDNN=False
        self.MKLDNNMODEL=False

    def train(self, dataset, num_epochs, loss_fn, val_dataset=None, val_metric_fn=None, initfrombest=False, pool=None, nocpu=False, rank=0, size=1):
        """
        training the neural network

        Args:
            dataset (pytorch.DataLoader): iterator of the training dataset
            num_epoches (int): maximum number of epch for traning 
            loss_fn (callable): define the loss function 
            val_dataset (pytorch.DataLoader): iterator of the test dataset 
            val_metric_fn (callabel): define the loss function for validation dataset
            initfrombest (bool): if true, the training will start from the previous best fit point 
            pool (not used): reserved for parallel training 
            nocpu (bool): if true: use gpu to train, else: use cpu to traing
            rank (not used): reserved for parallel training
            size (not used): reserved for parallel training

        Returns:
            np.array: training losses in each epoch 
            np.array: validation losses in each epoch
        """
        torch.manual_seed(1234) 
        if self.optim == "automatic":
            if (rank==0) and (not os.path.isfile(self.outdir+"/lr.npy")):
                newmodel = copy.deepcopy(self.model)
                optimizer = torch.optim.AdamW(newmodel.parameters(), lr=1E-4, weight_decay=1E-4)
                lr_finder = LRFinder(newmodel, optimizer, loss_fn, device=self.device)
                lr_finder.range_test(dataset, val_loader=val_dataset, end_lr=5E-3, num_iter=100)
                fig = plt.figure()
                lr_finder.plot(log_lr=True)
                plt.savefig(os.path.join(self.outdir, "lr_tunning.png"))
                plt.close()
                lrs = lr_finder.history["lr"]
                losses = lr_finder.history["loss"]
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
                lr = lrs[min_grad_idx]
                if lr>1E0:
                   lr = lr/1E2
                np.save(self.outdir+"/lr.npy", lr)
            else:
                while(True):
                    try:
                        lr = np.load(self.outdir+"/lr.npy")
                        break
                    except:
                        pass
        lr = lr*size
        if initfrombest:
            ischeckpoint = self.load_checkpoint()
            if not ischeckpoint:
                print("best.pth.tar does not exsit")
        train_losses = []
        val_metrics = []

        if val_metric_fn is None:
            val_metric_fn = loss_fn
        es = EarlyStopping(patience=500)
        if rank==0:
            pbar = tqdm(range(num_epochs))
        else:
            pbar = range(num_epochs)

        old = 0 
        told = 0
        self.model_old = copy.deepcopy(self.model)
        if pool is not None:
            self.model = DDP(self.model_old)
        self.optim = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1E-4)
        for i in pbar:
            data_iter = iter(dataset)

            self.model.train()

            for X, y_target in data_iter:
                X = X.to(self.device)
                y_target = y_target.to(self.device)

                self.optim.zero_grad()
                y_pred = self.model(self.X_transform(X))
                if pool is not None:
                        loss = loss_fn(y_pred, y_target)
                        all_loss = [torch.zeros_like(loss) for _ in range(size)]
                        loss.backward()
                else:
                    loss = loss_fn(y_pred, y_target) 
                    loss.backward()

                self.optim.step()
                train_losses.append(loss.item())
            if val_dataset is not None:
                val_iter = iter(val_dataset)
                self.model.eval()
                modelmkl = copy.deepcopy(self.model)
                if not nocpu:
                    modelmkl = mkldnn_utils.to_mkldnn(modelmkl)

                val_metric = None
                val_count = 0
                for X, y_target in val_iter:
                    X = X.to(self.device)
                    y_target = y_target.to(self.device)

                    with torch.no_grad():
                        y_pred = modelmkl(self.X_transform(X))

                        if val_metric is None:
                            val_metric = np.array([val.item() for val in val_metric_fn(y_pred, y_target)])
                        else:
                            val_metric += np.array([val.item() for val in val_metric_fn(y_pred, y_target)])

                        val_count += 1

                val_metrics.append(val_metric / val_count)
                if rank==0:
                    pbar.set_description('Train/val Loss: {0:.5e}, {1:.5e}    Epoch'.format(loss, val_metrics[-1][0]))
                if self.outdir is not None:
                    is_best = val_metrics[-1][0] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics[-1][0]
                if ((np.std(np.array(val_metrics)[-10:,0])<0.01*(np.mean(np.array(val_metrics)[-10:,0])) and (i>=10) and (i<120) and (i%10==0))):
                    print("bad trainning: {0}".format(i), flush=True)
                    for ind, param_group in enumerate(self.optim.param_groups):
                        lr = param_group['lr']
                    self.model_old.init_weight()
                    del self.model
                    if pool is not None:
                        self.model = DDP(self.model_old)
                    else:
                        self.model = self.model_old
                    self.optim = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1E-4)
                    if ((i>10)&(lr>2E-4)):
                        for ind, param_group in enumerate(self.optim.param_groups):
                            lr = param_group['lr']
                            if lr>2E-6:
                                print("learning rate too large: {0}".format(lr), flush=True)
                                self.optim.param_groups[ind]['lr'] = lr/2.



                if (np.isnan(val_metrics[-1][0])) or (val_metrics[-1][0]>1E10) or ((val_metrics[-1][0]-old>5*old) and (i!=0)) or ((loss-told>5*told) and (i!=0)) :
                    #print( (val_metrics[-1][0]-old>5*old), ((loss-told>5*told)), flush=True)
                    for ind, param_group in enumerate(self.optim.param_groups):
                        lr = param_group['lr']
                    #self.model = self.model_old
                    ischeckpoint = self.load_checkpoint(ismpi=size>1)
                        #if pool is not None:
                            #self.model = DDP(self.model)
                            #state = PowerSGDState(process_group=None, matrix_approximation_rank=1)
                            #self.model.register_comm_hook(state, batched_powerSGD_hook)
                    if (not ischeckpoint):
                        self.model_old.init_weight()
                        del self.model
                        if pool is not None:
                            self.model = DDP(self.model_old)
                            #state = PowerSGDState(process_group=None, matrix_approximation_rank=1)
                            #self.model.register_comm_hook(state, batched_powerSGD_hook)
                        else:
                            self.model = self.model_old
                    self.optim = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1E-4)
                    if (np.isnan(val_metrics[-1][0])) or (val_metrics[-1][0]>1E10) or (val_metrics[-1][0]-old>10*old):
                        if (i>10):
                            for ind, param_group in enumerate(self.optim.param_groups):
                                lr = param_group['lr']
                                if lr>2E-6:
                                    print("learning rate too large: {0}".format(lr), flush=True)
                                    self.optim.param_groups[ind]['lr'] = lr/2.
                    if not (np.isnan(val_metrics[-1][0])):
                        if (val_metrics[-1][0]-old>5*old):
                            val_metrics[-1][0] = old
                    #else:
                    #    for param_group in self.optim.param_groups:
                    #        lr = param_group['lr']
                    #        print("learning rate too large: {0}".format(lr), flush=True)
                    #        param_group['lr'] = lr/2.
                else: 
                    criteria = es.step(val_metrics[-1][0], loss)
                    if criteria == 1:
                        for ind, param_group in enumerate(self.optim.param_groups):
                            lr = param_group['lr']
                            wd = param_group['weight_decay']
                            if lr>2E-6:
                                print("\n learning rate too large: {0}\n".format(lr), flush=True)
                                self.optim.param_groups[ind]['lr'] = lr/2.
                                self.optim.param_groups[ind]['weight_decay'] = wd/2
                            else:
                                es.cooling=0
                        
                    if criteria ==2:
                        print("early stop", flush=True)
                        for ind, param_group in enumerate(self.optim.param_groups):
                            lr = param_group['lr']
                            print("learning rate", lr, flush=True)
                        if rank==0:
                            break
                    if criteria ==3:
                        for ind, param_group in enumerate(self.optim.param_groups):
                            wd = param_group['weight_decay']
                            print("\n weight decay too small: {0}\n".format(wd), flush=True)
                            if wd<1E0:
                                self.optim.param_groups[ind]['weight_decay'] = wd*2
                            #else:
                            #    es.cooling_weight_decay=0
                old = val_metrics[-1][0]
                told = loss
                        
            if (self.outdir is not None)&(rank==0):
                if pool is not None:
                    save_checkpoint({'epoch': i + 1,
                               'state_dict': self.model.module.state_dict(),
                               'optim_dict' : self.optim.state_dict(),
                               'mpi_state_dict': self.model.state_dict()
                               },
                               is_best=is_best,
                               checkpoint=self.outdir)
                else:
                    save_checkpoint({'epoch': i + 1,
                               'state_dict': self.model.state_dict(),
                               'optim_dict' : self.optim.state_dict()},
                               is_best=is_best,
                               checkpoint=self.outdir)


                if (i%100)==0: 
                    train_loss = np.array(train_losses)
                    val_metric = np.array(val_metrics)
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
                    plt.savefig(os.path.join(self.outdir, "training_progress.png"))
                    plt.close()



        if val_dataset is not None:
            return np.array(train_losses), np.array(val_metrics)
        else:
            return np.array(train_losses)

    def load_checkpoint(self, ismpi=False):
        """
        internal use only
        """
        if os.path.isfile(os.path.join(self.outdir, "best.pth.tar")):
            load_checkpoint(os.path.join(self.outdir, "best.pth.tar"), self.model, self.optim, device=self.device, ismpi=ismpi)
            return True
        else:
            return False

    def predict(self, X, no_grad=True):
        """
        make model evaluation 

        Args:
            X (torch tensor): parameters 
            no_grad (bool): True: not keep gradient information, Flase: keep gradient information

        Returns:
            torch.tensor: model evaluation 
        """
        self.model.eval()

        if (len(X.shape) == 1):
            X = X.view(1, -1)
            one_input = True
        else:
            one_input = False
        X_new = self.X_transform(X)
        if self.MKLDNN:
            X_new = X_new.to_mkldnn()
            if not self.MKLDNNMODEL:
                self.MKLDNNmodel = torch.jit.optimize_for_inference(torch.jit.script(mkldnn_utils.to_mkldnn(self.model)))
                self.MKLDNNMODEL=True
            
            if no_grad:
                with torch.no_grad():
                    y_pred = self.MKLDNNmodel(X_new)
            else:
                y_pred = self.MKLDNNmodel(X_new)
        else:
            if no_grad:
                with torch.no_grad():
                    y_pred = self.model(X_new)
            else:
                y_pred = self.model(X_new)

        if self.MKLDNN:
            y_pred = y_pred.to_dense()   
        y_pred = self.y_transform(y_pred)

        if one_input:
            y_pred = y_pred.view(-1)
        return y_pred

