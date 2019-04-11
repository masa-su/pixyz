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
        self.DistributionTorch = None

        super().__init__(var=var, cond_var=[], **kwargs)

    @property
    def input_var(self):
        """
        In CustomLikelihoodDistribution, `input_var` is same as `var`.
        """

        return self.var

    @property
    def distribution_name(self):
        return "Custom Distribution"

    def log_likelihood(self, x_dict):

        if not set(list(x_dict.keys())) >= set(self._var):
            raise ValueError("Input's keys are not valid.")

        _x_dict = get_dict_values(x_dict, self._var)
        log_like = torch.log(self.likelihood(_x_dict[0]))
        # log_like = sum_samples(log_like)
        return log_like
