from .losses import SetLoss


class ELBO(SetLoss):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64]) # q(z|x)
    >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"], features_shape=[64]) # p(x|z)
    >>> loss_cls = ELBO(p, q)
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) - \log p(z|x) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    """
    def __init__(self, p, q, input_var=None):

        loss = (p.log_prob() - q.log_prob()).expectation(q, input_var)
        super().__init__(loss)
