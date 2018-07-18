from __future__ import print_function
import torch
from torch import nn
from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import Categorical as CategoricalTorch

from ..utils import get_dict_values
from .operators import MultiplyDistribution


class Distribution(nn.Module):

    def __init__(self, cond_var=[], var=["x"], dim=1,
                 **kwargs):
        super(Distribution, self).__init__()
        self.cond_var = cond_var
        self.var = var
        self.dim = dim  # default: 1

        if len(cond_var) == 0:
            self.prob_text = "p({})".format(
                ','.join(var)
            )
        else:
            self.prob_text = "p({}|{})".format(
                ','.join(var),
                ','.join(cond_var)
            )
        self.prob_factorized_text = self.prob_text

        # whether I'm a distribution with constant parameters
        self.dist = None

        self.constant_params = {}
        self.map_dict = {}

        for keys in self.params_keys:
            if keys in kwargs.keys():
                if type(kwargs[keys]) is str:
                    self.map_dict[kwargs[keys]] = keys
                else:
                    self.constant_params[keys] = kwargs[keys]

        # Set the distribution if all parameters are constant and
        # set at initialization.
        if len(self.constant_params) == len(self.params_keys):
            self._set_distribution()

    def _set_distribution(self, x={}):
        params = self.get_params(**x)
        self.dist = self.DistributionTorch(**params)

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

    def _verify_input(self, x, var=None):
        # To verify whether input is valid.
        # input : tensor, list or dict
        # output : dict

        if var is None:
            var = self.cond_var

        if type(x) is torch.Tensor:
            x = {var[0]: x}

        elif type(x) is list:
            x = dict(zip(var, x))

        elif type(x) is dict:
            if not set(list(x.keys())) == set(var):
                raise ValueError("Input's keys are not valid.")

        else:
            raise ValueError("The type of input is not valid, got %s."
                             % type(x))

        return x

    def get_params(self, **params):
        """
        Examples
        --------
        >> > print(dist_1.prob_text, dist_1.distribution_name)
        >> > p(x) Normal
        >> > dist_1.get_params()
        >> > {"loc": 0, "scale": 1}
        >> > print(dist_2.prob_text, dist_2.distribution_name)
        >> > p(x|z) Normal
        >> > dist_1.get_params({"z": 1})
        >> > {"loc": 0, "scale": 1}
        """

        output = self.forward(**params)

        # append constant_params to map_dict
        output.update(self.constant_params)

        return output

    def sample(self, x=None, shape=None, batch_size=1, return_all=True,
               reparam=True):
        # input : tensor, list or dict
        # output : dict

        if x is None:  # unconditional
            if len(self.cond_var) != 0:
                raise ValueError("You should set inputs or parameters")

            if shape:
                sample_shape = shape
            else:
                sample_shape = (batch_size, self.dim)

            output =\
                {self.var[0]: self._get_sample(reparam=reparam,
                                               sample_shape=sample_shape)}

        else:  # conditional
            x = self._verify_input(x)
            self._set_distribution(x)

            output = {self.var[0]: self._get_sample(reparam=reparam)}

            if return_all:
                output.update(x)

        return output

    def log_likelihood(self, x):
        # input : dict
        # output : dict

        if not set(list(x.keys())) >= set(self.cond_var + self.var):
            raise ValueError("Input's keys are not valid.")

        if len(self.cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self.cond_var, True)
            self._set_distribution(_x)

        log_like = self._get_log_like(x)
        return mean_sum_samples(log_like)

    def forward(self, **x):
        """
        Examples
        --------
        >> > distribution.map_dict
        >> > {"a": "loc"}
        >> > x = {"a": 0}
        >> > distribution.forward(x)
        >> > {"loc": 0}
        """

        output = {self.map_dict[key]: value for key, value in x.items()}
        return output

    def sample_mean(self):
        NotImplementedError

    def __mul__(self, other):
        return MultiplyDistribution(self, other)


class CustomLikelihoodDistribution(Distribution):

    def __init__(self, var=["x"],  likelihood=None,
                 **kwargs):
        if likelihood is None:
            raise ValueError("You should set the likelihood"
                             " of this distribution.")
        self.likelihood = likelihood
        self.params_keys = []
        self.distribution_name = "Custom Distribution"
        self.DistributionTorch = None

        super(CustomLikelihoodDistribution,
              self).__init__(var=var, cond_var=[], **kwargs)

    def _set_distribution(self, x={}):
        pass

    def _get_log_like(self, x):
        # input : dict
        # output : tensor

        x_targets = get_dict_values(x, self.var)
        return torch.log(self.likelihood(x_targets[0]))

    def get_params(self, **kwargs):
        pass

    def sample(self, **kwargs):
        pass


class Normal(Distribution):

    def __init__(self, **kwargs):
        self.params_keys = ["loc", "scale"]
        self.distribution_name = "Normal"
        self.DistributionTorch = NormalTorch

        super(Normal, self).__init__(**kwargs)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["loc"]


class Bernoulli(Distribution):

    def __init__(self, *args, **kwargs):
        self.params_keys = ["probs"]
        self.distribution_name = "Bernoulli"
        self.DistributionTorch = BernoulliTorch

        super(Bernoulli, self).__init__(*args, **kwargs)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class Categorical(Distribution):

    def __init__(self, one_hot=True, *args, **kwargs):
        self.one_hot = one_hot
        self.params_keys = ["probs"]
        self.distribution_name = "Categorical"
        self.DistributionTorch = CategoricalTorch
        # TODO: use OneHotCategorical

        super(Categorical, self).__init__(*args, **kwargs)

    def _get_sample(self, *args, **kwargs):
        samples = super(Categorical,
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

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


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
