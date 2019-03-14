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

    def __init__(self, p, q, input_var=None, dim=None):
        self.dim = dim
        super().__init__(p, q, input_var)

    @property
    def loss_text(self):
        return "KL[{}||{}]".format(self._p.prob_text, self._q.prob_text)

    def _get_estimated_value(self, x, **kwargs):
        if (isinstance(self._p, DistributionBase) is False) or (isinstance(self._q, DistributionBase) is False):
            raise ValueError("Divergence between these two distributions cannot be estimated, "
                             "got %s and %s." % (self._p.distribution_name, self._q.distribution_name))

        inputs = get_dict_values(x, self._p.input_var, True)
        self._p.set_distribution(inputs)

        inputs = get_dict_values(x, self._q.input_var, True)
        self._q.set_distribution(inputs)

        divergence = kl_divergence(self._p.dist, self._q.dist)

        if self.dim:
            _kl = torch.sum(divergence, dim=self.dim)
            return divergence, x

        dim_list = list(torch.arange(divergence.dim()))
        divergence = torch.sum(divergence, dim=dim_list[1:])
        return divergence, x
