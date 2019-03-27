from .expectations import Expectation


class ELBO(Expectation):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the `Expectation` class.
    """
    def __init__(self, p, q, input_var=None):
        if input_var is None:
            input_var = q.input_var

        super().__init__(q, p.log_prob() - q.log_prob(), input_var)
