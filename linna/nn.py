from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import torch
import pickle


class ResBlock_batchnorm(nn.Module):
    """
    Residual block
    """
    def __init__(self, in_size, channel, out_size):
        """
        Args:
            in_size (int): size of input data vector
            channel (int): size of the inner layer
            out_size (int): size of output data vector

        """
        super(ResBlock_batchnorm, self).__init__()

        self.layer1 = nn.Linear(in_size, channel)
        self.layer2 = nn.Linear(channel, out_size)

        if in_size == out_size:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Linear(in_size, out_size, bias=False)

        self.init_weight()
    def init_weight(self):
        """
        initialize the weight of neural network
        """
        for m in self.modules():
            if type(m)==nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1E-2)
        nn.init.zeros_(self.skip_layer.weight)

    def forward(self, x):
        """
        Args:
            s (torch tensor): input array
        Returns:
            torch tensor: ourput array

        """
        h = F.relu((self.layer1(x)))
        y = F.relu((self.layer2(h)) * 0.1 + self.skip_layer(x))

        return y


class ChtoModelv2(nn.Module):
    """
        main neural network used by linna
    """
    def __init__(self, in_size, out_size, linearmodel, docpu=False): 
        """
    
        Args:
            in_size (int): size of input data vector
            out_size (int): size of output data vector
            linear model (function): input: tensor array, output: tensor array. One might want to add a pretrained model on top of the NN. 
            docpu (bool) whether to use cpu for evaluation
        """ 
        super(ChtoModelv2, self).__init__()
        self.channel = 16
        hidden_size = max(32, int(out_size*32))
        if out_size>30:
            hidden_size=1000
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = ResBlock_batchnorm(hidden_size, self.channel, hidden_size//2) 
        hidden_size = hidden_size//2
        self.layer3 = ResBlock_batchnorm(hidden_size, int(self.channel*2), hidden_size//2) 
        hidden_size = hidden_size//2
        self.layer4 = ResBlock_batchnorm(hidden_size, int(self.channel*4), hidden_size//2) 
        hidden_size = hidden_size//2
        self.layer6 = nn.Linear(hidden_size, hidden_size*4) 
        self.layer7 = nn.Linear(hidden_size*4, out_size) 
        self.layer8 = nn.Linear(out_size, out_size) 
        self.init_weight()
        self.linearmodel = linearmodel
        self.docpu = docpu

    def init_weight(self):
        """
            initialize weights for neural network
        """
        for m in self.modules():
           if type(m)==nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1E-2)
           elif type(m)==ResBlock_batchnorm:
                m.init_weight()
           elif type(m)==ChtoModelv2:
                pass
           elif type(m)==nn.modules.batchnorm.BatchNorm1d:
                pass
           else:
               print(type(m), flush=True)
               assert(0)
            
    def forward(self, s):
        """

        Args:
            s (torch tensor): input array
        Returns:
            torch tensor: ourput array

        """
        if self.docpu:
            s = s.to_mkldnn()
        s_in  = F.relu(self.layer1(s)) 
        s_in  = self.layer2(s_in) 
        s_in  = self.layer3(s_in) 
        s_in  = self.layer4(s_in) 
        s_in  = F.relu(self.layer6(s_in))
        s_in  = F.relu(self.layer7(s_in))
        if self.linearmodel is not None:
            s = self.layer8(s_in)+self.linearmodel(s)
        else:
            s = self.layer8(s_in)
        if self.docpu:
           s = s.to_dense()
        return s


class ChtoModelv2_linear(nn.Module):
    """
    For testing 
    """
    def __init__(self, in_size, out_size, linearmodel, docpu=False): 
        super(ChtoModelv2_linear, self).__init__()
        self.channel = 16
        hidden_size = max(32, int(out_size*32))
        if out_size>30:
            hidden_size=1000
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = ResBlock_batchnorm(hidden_size, self.channel, hidden_size//2) 
        hidden_size = hidden_size//2
        self.layer3 = ResBlock_batchnorm(hidden_size, int(self.channel*2), hidden_size//2) 
        hidden_size = hidden_size//2
        self.layer4 = ResBlock_batchnorm(hidden_size, int(self.channel*4), hidden_size//2) 
        hidden_size = hidden_size//2
        #self.layer5 = ResBlock_batchnorm(hidden_size, self.channel*8, hidden_size//2) 
        #hidden_size = hidden_size//2
        #self.layer6 = ResBlock_batchnorm(hidden_size, self.channel*8, hidden_size//2) 
        #hidden_size = hidden_size//2
        self.layer6 = nn.Linear(hidden_size, hidden_size*4) 
        self.layer7 = nn.Linear(hidden_size*4, out_size) 
        self.layer8 = nn.Linear(out_size, out_size) 
        self.linearlayer  = nn.Linear(in_size, out_size) 
        self.init_weight()
        self.linearlayer.bias.data.fill_(0)
        self.linearlayer.weight.data.fill_(1E-5)
        self.linearmodel = linearmodel
        #if not linearmodel.istrained():
        #    raise(ValueError("linear model must be trained before pass it to ChtoModel_v2"))
        self.docpu = docpu

    def init_weight(self):
        for m in self.modules():
           if type(m)==nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1E-2)
           elif type(m)==ResBlock_batchnorm:
                m.init_weight()
           elif type(m)==ChtoModelv2_linear:
                pass
           elif type(m)==nn.modules.batchnorm.BatchNorm1d:
                pass
           else:
               print(type(m), flush=True)
               assert(0)
            
    def forward(self, s):
        if self.docpu:
            s = s.to_mkldnn()
        s_in  = F.relu(self.layer1(s)) 
        s_in  = self.layer2(s_in) 
        s_in  = self.layer3(s_in) 
        s_in  = self.layer4(s_in) 
        #s_in  = self.layer5(s_in)
        s_in  = F.relu(self.layer6(s_in))
        s_in  = F.relu(self.layer7(s_in))
        s = self.layer8(s_in) + 1E-3*self.linearlayer(s) 
        if self.docpu:
           s = s.to_dense()
        return s

class LinearModel:
    """
    Linear regression model
    """
    def __init__(self, norder, npc, x_transform=None, y_transform=None, y_inverse_transform_data=None):
        self._istrained=False
        self.norder = norder
        self.npc = npc 
        self.model_poly = None
        self.pcs = None
        self.vec = None 
        self.xmean = None
        self.xstd = None
        self.ymean = None
        self.ystd = None
        if x_transform is None:
            self.x_transform = lambda x: x
        else:
            self.x_transform = x_transform
        if y_transform is None:
            self.y_transform = lambda x:x
        else:
            self.y_transform = y_transform
        if y_inverse_transform_data is None:
            self.y_invtransform_data = lambda x:x
        else:
            self.y_invtransform_data = y_inverse_transform_data
    def train(self, train_x, train_y, sample_weight=None):
        #Do PCA
        self.xmean = torch.mean(train_x, axis=0)
        self.ymean = torch.mean(train_y, axis=0)
        self.xstd = torch.std(train_x, axis=0)
        self.ystd = torch.std(train_y, axis=0)
        train_x_norm = (train_x-self.xmean)/self.xstd
        train_y_norm = (train_y-self.ymean)/self.ystd
        try:
            vec,pcs,  _= torch.linalg.svd(train_y_norm.T.matmul(train_y_norm))
        except:
            L = train_y_norm.T.matmul(train_y_norm)
            print(L.mean())
            vec,pcs,  _= torch.linalg.svd(L + 1e-4*L.mean()*torch.rand(L.shape[0],L.shape[1]))
        if self.npc is None:
            self.npc = np.where(pcs/pcs[0]>0.05)[0][-1]+1
        train_y_projected = train_y_norm.matmul(vec)[:, :self.npc]
        self.vec = vec[:,:self.npc].T
        self.pcs = pcs
        #Train linear model
        self.model_poly = pytorchPolynomialLinear(self.norder)
        self.model_poly.fit(train_x_norm, train_y_projected, sample_weight=sample_weight)
        
        self._istrained=True
    def __call__(self,x):
        xnorm = (x-self.xmean)/self.xstd
        if len(x.shape)==1:
            return self.model_poly(xnorm.reshape(1,-1)).matmul(self.vec)*self.ystd+self.ymean
        else:
            return self.model_poly(xnorm).matmul(self.vec)*self.ystd+self.ymean
        
    def istrained(self):
        return self._istrained

    def save(self, outname):
        f = open(outname, 'wb')
        pickle.dump(self,f)
        f.close()

    def predict(self,x):
        return self.y_invtransform_data(self.y_transform(self(self.x_transform(x))))
        
class pytorchPolynomialLinear:
    """
    Implement a polynomial fit using pytorch

    """
    def __init__(self, ndegree):
        self.poly =  PolynomialFeatures(degree=ndegree)

        self.model_pol = Pipeline([('poly', self.poly),
                  ('linear', LinearRegression(fit_intercept=False))])
        self.coef = None
    def fit(self, train_x, train_y, sample_weight=None):
        kwargs = {self.model_pol.steps[-1][0] + '__sample_weight': sample_weight}
        model_pol = self.model_pol.fit(train_x, train_y, **kwargs)
        self.coef = self.model_pol.named_steps['linear'].coef_

    def trainsform(self, tensor):
        if len(tensor.shape)==1:
            inshape=1
        else:
            inshape = tensor.shape[0]
        result = torch.zeros(size=(inshape, self.poly.n_output_features_))
        powers = self.poly.powers_
        powers = torch.tensor(powers)
        for i, power in enumerate(powers):
            result[:, i] = torch.prod(tensor**power,axis=1)
        return result

    def __call__(self, X):
        return torch.matmul(self.trainsform(X),torch.from_numpy(self.coef.T.astype(np.float32)).to('cpu').requires_grad_())


