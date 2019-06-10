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
    r"""
    Reconstruction Loss (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x|z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64], name="q") # q(z|x)
    >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"], features_shape=[64], name="p") # p(x|z)
    >>> loss_cls = StochasticReconstructionLoss(q, p)
    >>> print(loss_cls)
    - \mathbb{E}_{q(z|x)} \left[\log p(x|z) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1,64)})
    """

    def __init__(self, encoder, decoder, input_var=None):

        if input_var is None:
            input_var = encoder.input_var

        if not(set(decoder.var) <= set(input_var)):
            raise ValueError("Variable {} (in the `{}` class) is not included"
                             " in `input_var` of the `{}` class.".format(decoder.var,
                                                                         decoder.__class__.__name__,
                                                                         encoder.__class__.__name__))

        loss = -decoder.log_prob().expectation(encoder, input_var)
        super().__init__(loss)
