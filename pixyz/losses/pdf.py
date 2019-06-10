import sympy
import torch
from .losses import Loss


class LogProb(Loss):
    r"""
    The log probability density/mass function.

    .. math::

        \log p(x)

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...            features_shape=[10])
    >>> loss_cls = LogProb(p)  # or p.log_prob()
    >>> print(loss_cls)
    \log p(x)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([12.9894, 15.5280])

    """

    def __init__(self, p, sum_features=True, feature_dims=None):
        input_var = p.var + p.cond_var
        self.sum_features = sum_features
        self.feature_dims = feature_dims
        super().__init__(p, input_var=input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("\\log {}".format(self.p.prob_text))

    def _get_eval(self, x={}, **kwargs):
        log_prob = self.p.get_log_prob(x, sum_features=self.sum_features, feature_dims=self.feature_dims)
        return log_prob, x


class Prob(LogProb):
    r"""
    The probability density/mass function.

    .. math::

        p(x) = \exp(\log p(x))

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...            features_shape=[10])
    >>> loss_cls = Prob(p)  # or p.prob()
    >>> print(loss_cls)
    p(x)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([3.2903e-07, 5.5530e-07])
    """

    @property
    def _symbol(self):
        return sympy.Symbol(self.p.prob_text)

    def _get_eval(self, x={}, **kwargs):
        log_prob, x = super()._get_eval(x, **kwargs)
        return torch.exp(log_prob), x
