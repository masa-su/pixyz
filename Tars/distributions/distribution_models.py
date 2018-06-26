from __future__ import print_function
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli, Categorical

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
            self.prob_text = "p(" + ','.join(var) + "|"\
                             + ','.join(cond_var) + ")"
        self.prob_factorized_text = self.prob_text

        self.dist = None  # whether I'm a deep distribution or not
        self.dim = dim  # default: 1

    def _set_dist(self):
        NotImplementedError

    def _get_sample(self, reparam=True,
                    sample_shape=torch.Size()):

        if reparam:
            try:
                return self.dist.rsample(sample_shape=sample_shape)
            except NotImplementedError:
                print("We can not use the reparameterization trick"
                      "for this distribution.")

        return self.dist.sample(sample_shape=sample_shape)

    def _get_log_like(self, x):
        # input : dict
        # output : tensor

        x_targets = get_dict_values(x, self.var)
        return self.dist.log_prob(*x_targets)

    def _get_forward(self, x):
        # input : dict
        # output : tensor

        x_inputs = get_dict_values(x, self.cond_var)
        return self.forward(*x_inputs)

    def sample(self, x=None, shape=None, batch_size=1, return_all=True,
               reparam=True):
        # input : tensor, list or dict
        # output : dict

        if (len(self.cond_var) == 0) and (x is None):  # unconditional
            if shape:
                sample_shape = shape
            else:
                sample_shape = (batch_size, self.dim)

            output =\
                {self.var[0]: self._get_sample(reparam=reparam,
                                               sample_shape=sample_shape)}

        elif x is not None:  # conditional
            if type(x) is torch.Tensor:
                x = {self.cond_var[0]: x}

            elif type(x) is list:
                x = dict(zip(self.cond_var, x))

            elif type(x) is dict:
                if not set(list(x.keys())) == set(self.cond_var):
                    raise ValueError("Input's keys are not valid.")

            else:
                raise ValueError("Invalid input")

            params = self._get_forward(x)
            self._set_dist(params)

            output = {self.var[0]: self._get_sample(reparam=reparam)}

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

        if len(self.cond_var) > 0:  # conditional distribution
            params = self._get_forward(x)
            self._set_dist(params)

        log_like = self._get_log_like(x)
        return mean_sum_samples(log_like)

    def __mul__(self, other):
        return MultiplyDistributionModel(self, other)

    def forward(self, *input):
        return super(DistributionModel, self).__init__(*input)


class NormalModel(DistributionModel):

    def __init__(self, loc=None, scale=None, *args, **kwargs):
        super(NormalModel, self).__init__(*args, **kwargs)

        if (loc is not None) and (scale is not None):
            self._set_dist([loc, scale])
            self.mu = loc
            self.sigma = scale
        self.distribution_name = "Normal"

    def _set_dist(self, params):
        [loc, scale] = params
        self.dist = Normal(loc=loc, scale=scale)

    def sample_mean(self, x):
        mu, _ = self._get_forward(x)
        return mu


class BernoulliModel(DistributionModel):

    def __init__(self, probs=None, *args, **kwargs):
        super(BernoulliModel, self).__init__(*args, **kwargs)

        if probs:
            self._set_dist(probs)
            self.probs = probs
        self.distribution_name = "Bernoulli"

    def _set_dist(self, probs):
        self.dist = Bernoulli(probs=probs)

    def sample_mean(self, x):
        mu = self._get_forward(x)
        return mu


class CategoricalModel(DistributionModel):

    def __init__(self, probs=None, one_hot=True, *args, **kwargs):
        super(CategoricalModel, self).__init__(*args, **kwargs)

        self.one_hot = one_hot
        if probs:
            self._set_dist(probs)
            self.probs = probs
        self.distribution_name = "Categorical"

    def _get_sample(self, *args, **kwargs):
        samples = super(CategoricalModel,
                        self)._get_sample(*args, **kwargs)

        if self.one_hot:
            # convert to one-hot vectors
            samples = torch.eye(self.dist._num_events)[samples]

        return samples

    def _get_log_like(self, x, *args, **kwargs):
        [x_target] = get_dict_values(x, self.var)

        # for one-hot representation
        x_target = torch.argmax(x_target, dim=1)

        return self.dist.log_prob(x_target)

    def _set_dist(self, probs):
        self.dist = Categorical(probs=probs)

    def sample_mean(self, x):
        mu = self._get_forward(x)
        return mu


def mean_sum_samples(samples):
    dim = samples.dim()
    if dim == 4:
        return torch.mean(torch.sum(torch.sum(samples, dim=2), dim=2), dim=1)
    elif dim == 3:
        return torch.sum(torch.sum(samples, dim=-1), dim=-1)
    elif dim == 2:
        return torch.sum(samples, dim=-1)
    elif dim == 1:
        return samples
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % dim)
