import sympy

from .losses import Loss, SetLoss


class Entropy(SetLoss):
    r"""
    Entropy (Monte Carlo approximation).

    .. math::

        H[p] = -\mathbb{E}_{p(x)}[\log p(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64])
    >>> loss_cls = Entropy(p)
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log p(x) \right]
    >>> loss = loss_cls.eval()
    """

    def __init__(self, p, input_var=None):
        if input_var is None:
            input_var = p.input_var

        loss = -p.log_prob().expectation(p, input_var)
        super().__init__(loss)


class AnalyticalEntropy(Loss):
    r"""
    Entropy (analytical).

    .. math::

        H[p] = -\mathbb{E}_{p(x)}[\log p(x)]

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64])
    >>> loss_cls = AnalyticalEntropy(p)
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log p(x) \right]
    >>> loss = loss_cls.eval()
    """

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol("- \\mathbb{{E}}_{} \\left[{} \\right]".format(p_text, self.p.log_prob().loss_text))

    def _get_eval(self, x_dict, **kwargs):
        if not hasattr(self.p, 'distribution_torch_class'):
            raise ValueError("Entropy of this distribution cannot be evaluated, "
                             "got %s." % self.p.distribution_name)

        entropy = self.p.get_entropy(x_dict)

        return entropy, x_dict


class CrossEntropy(SetLoss):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation).

    .. math::

        H[p||q] = -\mathbb{E}_{p(x)}[\log q(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log q(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64], name="p")
    >>> q = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"], features_shape=[64], name="q")
    >>> loss_cls = CrossEntropy(p, q)
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log q(x) \right]
    >>> loss = loss_cls.eval()
    """

    def __init__(self, p, q, input_var=None):
        if input_var is None:
            input_var = list(set(p.input_var + q.input_var) - set(p.var))

        loss = -q.log_prob().expectation(p, input_var)
        super().__init__(loss)


class StochasticReconstructionLoss(SetLoss):
    def __init__(self, encoder, decoder, input_var=None):
        raise NotImplementedError("This function is obsolete."
                                  " please use `-decoder.log_prob().expectation(encoder)` instead of it.")
