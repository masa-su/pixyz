import sympy
import torch
from .losses import Loss


class LogProb(Loss):
    r"""
    The log probability density/mass function.

    .. math::

        \log p(x)
    """

    def __init__(self, p, sum_features=True, feature_dims=None):
        input_var = p.var + p.cond_var
        self.sum_features = sum_features
        self.feature_dims = feature_dims
        super().__init__(p, input_var=input_var)

    @property
    def _symbol(self):
        return sympy.log(sympy.Symbol(self._p.prob_text))

    def _get_eval(self, x={}, **kwargs):
        log_prob = self._p.get_log_prob(x, sum_features=self.sum_features, feature_dims=self.feature_dims)
        return log_prob, x


class Prob(LogProb):
    r"""
    The probability density/mass function.

    .. math::

        p(x) = \exp(\log p(x))
    """

    @property
    def _symbol(self):
        return sympy.Symbol(self._p.prob_text)

    def _get_eval(self, x={}, **kwargs):
        log_prob, x = super()._get_eval(x, **kwargs)
        return torch.exp(log_prob), x
