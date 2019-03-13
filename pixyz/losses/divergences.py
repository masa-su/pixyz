import torch
from torch.distributions import kl_divergence

from ..utils import get_dict_values
from ..distributions.distributions import DistributionBase
from .losses import Loss


class KullbackLeibler(Loss):
    r"""
    Kullback-Leibler divergence (analytical).

    .. math::

        D_{KL}[p||q] = \mathbb{E}_{p(x)}[\log \frac{p(x)}{q(x)}]

    TODO: This class seems to be slightly slower than this previous implementation
     (perhaps because of `set_distribution`).
    """

    def __init__(self, p1, p2, input_var=None, dim=None):
        self.dim = dim
        super().__init__(p1, p2, input_var)

    @property
    def loss_text(self):
        return "KL[{}||{}]".format(self._p1.prob_text, self._p2.prob_text)

    def _get_estimated_value(self, x, **kwargs):
        if (isinstance(self._p1, DistributionBase) is False) or (isinstance(self._p2, DistributionBase) is False):
            raise ValueError("Divergence between these two distributions cannot be estimated, "
                             "got %s and %s." % (self._p1.distribution_name, self._p2.distribution_name))

        inputs = get_dict_values(x, self._p1.input_var, True)
        self._p1.set_distribution(inputs)

        inputs = get_dict_values(x, self._p2.input_var, True)
        self._p2.set_distribution(inputs)

        divergence = kl_divergence(self._p1.dist, self._p2.dist)

        if self.dim:
            _kl = torch.sum(divergence, dim=self.dim)
            return divergence, x

        dim_list = list(torch.arange(divergence.dim()))
        divergence = torch.sum(divergence, dim=dim_list[1:])
        return divergence, x
