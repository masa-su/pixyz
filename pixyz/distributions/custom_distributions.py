import torch

from ..utils import get_dict_values
from .distributions import Distribution


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

        super().__init__(var=var, cond_var=[], **kwargs)

    def _set_distribution(self, x={}):
        pass

    def _get_log_like(self, x):
        # input : dict
        # output : tensor

        x_targets = get_dict_values(x, self._var)
        return torch.log(self.likelihood(x_targets[0]))

    def get_params(self, **kwargs):
        pass

    def sample(self, **kwargs):
        pass
