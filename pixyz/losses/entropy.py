import sympy
import torch

from pixyz.losses.losses import Loss
from pixyz.losses.divergences import KullbackLeibler


def Entropy(p, input_var=None, analytical=True, sample_shape=torch.Size([1])):
    r"""
    Entropy (Analytical or Monte Carlo approximation).

    .. math::

        H(p) &= -\mathbb{E}_{p(x)}[\log p(x)] \qquad \text{(analytical)}\\
        &\approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l), \quad \text{where} \quad x_l \sim p(x) \quad \text{(MC approximation)}.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64])
    >>> loss_cls = Entropy(p, analytical=True)
    >>> print(loss_cls)
    H \left[ {p(x)} \right]
    >>> loss_cls.eval()
    tensor([90.8121])
    >>> loss_cls = Entropy(p, analytical=False, sample_shape=[10])
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log p(x) \right]
    >>> loss_cls.eval() # doctest: +SKIP
    tensor([90.5991])
    """
    if analytical:
        loss = AnalyticalEntropy(p, input_var=input_var)
    else:
        loss = -p.log_prob().expectation(p, input_var, sample_shape=sample_shape)
    return loss


class AnalyticalEntropy(Loss):
    def __init__(self, p, input_var=None):
        if input_var is None:
            _input_var = p.input_var.copy()
        else:
            _input_var = list(input_var)
        super().__init__(_input_var)
        self.p = p

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol("H \\left[ {} \\right]".format(p_text))

    def forward(self, x_dict, **kwargs):
        if not hasattr(self.p, 'distribution_torch_class'):
            raise ValueError("Entropy of this distribution cannot be evaluated, "
                             "got %s." % self.p.distribution_name)

        entropy = self.p.get_entropy(x_dict)

        return entropy, {}


def CrossEntropy(p, q, input_var=None, analytical=False, sample_shape=torch.Size([1])):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation or Analytical).

    .. math::

        H(p,q) &= -\mathbb{E}_{p(x)}[\log q(x)] \qquad \text{(analytical)}\\
        &\approx -\frac{1}{L}\sum_{l=1}^L \log q(x_l), \quad \text{where} \quad x_l \sim p(x) \quad \text{(MC approximation)}.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64], name="p")
    >>> q = Normal(loc=torch.tensor(1.), scale=torch.tensor(1.), var=["x"], features_shape=[64], name="q")
    >>> loss_cls = CrossEntropy(p, q, analytical=True)
    >>> print(loss_cls)
    D_{KL} \left[p(x)||q(x) \right] + H \left[ {p(x)} \right]
    >>> loss_cls.eval()
    tensor([122.8121])
    >>> loss_cls = CrossEntropy(p, q, analytical=False, sample_shape=[10])
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log q(x) \right]
    >>> loss_cls.eval() # doctest: +SKIP
    tensor([123.2192])
    """
    if analytical:
        loss = Entropy(p) + KullbackLeibler(p, q)
    else:
        loss = -q.log_prob().expectation(p, input_var, sample_shape=sample_shape)
    return loss


class StochasticReconstructionLoss(Loss):
    def __init__(self, encoder, decoder, input_var=None, sample_shape=torch.Size([1])):
        raise NotImplementedError("This function is obsolete."
                                  " please use `-decoder.log_prob().expectation(encoder)` instead of it.")
