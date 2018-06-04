from __future__ import print_function
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli

from ..utils import get_dict_values
from .operators import MultiplyDistributionModel

class DistributionModel(nn.Module):
    
    def __init__(self, cond_var=[], var=["default_variable"], dim=1):
        super(DistributionModel, self).__init__()
        self.cond_var = cond_var
        self.var = var
        if len(cond_var) == 0:
            self.prob_text = "p(" + ','.join(var) + ")"
        else:
            self.prob_text = "p(" + ','.join(var) + "|" + ','.join(cond_var) + ")"
        self.prob_factorized_text = self.prob_text
        
        self.dist = None # whether I'm a deep distribution or not
        self.dim = dim # default: 1
    
    def _set_dist(self):
        NotImplementedError
    
    def sample(self, x=None, shape=None, batch_size=1, return_all=True):
        # input : tensor, list or dict
        # output : dict
        
        if (self.dist is not None) and (x is None):
            if shape:
                sample_shape = shape                
            else:
                sample_shape = (batch_size, self.dim)                
                        
            output = {self.var[0]: self.dist.rsample(sample_shape=sample_shape)}                
              
        elif x is not None:
            if type(x) is torch.Tensor:
                x = {self.cond_var[0]:x}
            
            elif type(x) is list:
                x = dict(zip(self.cond_var, x))
        
            elif type(x) is dict:
                if not set(list(x.keys())) == set(self.cond_var):
                    raise ValueError("Input's keys are not valid.")
                    
            else:
                raise ValueError("Invalid input")
                    
            x_inputs = get_dict_values(x, self.cond_var)
        
            params = self.forward(*x_inputs)
            dist = self._set_dist(params)
        
            output = {self.var[0]: dist.rsample()}
        
            if return_all:
                output.update(x)
        else:
            raise ValueError("You should set inputs or paramaters")
        
        return output
    
    def log_likelihood(self, x):
        # input : dict
        # output : dict
        
        if not set(list(x.keys())) == set(self.cond_var + self.var):
            raise ValueError("Input's keys are not valid.")
        
        if self.dist:
            x_targets = get_dict_values(x, self.var)
            log_like = self.dist.log_prob(*x_targets)
            
        else:
            x_inputs = get_dict_values(x, self.cond_var)
            params = self.forward(*x_inputs)
        
            dist = self._set_dist(params) 
            x_targets = get_dict_values(x, self.var)
            log_like = dist.log_prob(*x_targets)
        
        return mean_sum_samples(log_like)
    
    def __mul__(self, other):
        return MultiplyDistributionModel(self, other)
    
class GaussianModel(DistributionModel):

    def __init__(self, loc=None, scale=None, *args, **kwargs):
        super(GaussianModel, self).__init__(*args, **kwargs)

        if (loc is not None) and (scale is not None):
            self.distribution_name = "UnitGaussian"
            self.dist = self._set_dist([loc, scale])
        else:
            self.distribution_name = "Gaussian"
    
    def _set_dist(self, params):
        [loc, scale] = params
        dist = Normal(loc=loc, scale=scale)
        return dist

    def sample_mean(self, x):
        x_list = get_dict_values(x, self.cond_var)
        mu, _ = self.forward(*x_list)
        return mu
        
class BernoulliModel(DistributionModel):
    
    def __init__(self, probs=None, *args, **kwargs):
        super(BernoulliModel, self).__init__(*args, **kwargs)

        if probs:
            self.distribution_name = "UnitBernoulli"
            self.dist = self._set_dist(probs)
        else:
            self.distribution_name = "Bernoulli"
    
    def _set_dist(self, probs):
        dist = Bernoulli(probs=probs)
        return dist
    
    def sample_mean(self, x):
        x_list = get_dict_values(x, self.cond_var)
        mu = self.forward(*x_list)
        return mu

def mean_sum_samples(samples):
    dim = samples.dim()
    if dim == 4:
        return torch.mean(torch.sum(torch.sum(samples, dim=2), dim=2), dim=1)
    elif dim == 3:
        return torch.sum(torch.sum(samples, dim=-1), dim=-1)
    elif dim == 2:
        return torch.sum(samples, dim=-1)
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % dim)
