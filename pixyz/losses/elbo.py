from .losses import Loss


class ELBO(Loss):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),

    where :math:`z_l \sim q(z|x)`.
    """
    def __init__(self, p, approximate_dist, input_var=None):
        if input_var is None:
            input_var = approximate_dist.input_var

        super().__init__(p, approximate_dist, input_var=input_var)

    @property
    def loss_text(self):
        return "E_{}[log {}/{}]".format(self._p2.prob_text,
                                        self._p1.prob_text,
                                        self._p2.prob_text)

    def estimate(self, x={}, batch_size=None):
        _x = super().estimate(x)
        samples = self._p2.sample(_x, reparam=True, batch_size=batch_size)
        lower_bound = self._p1.log_likelihood(samples) -\
            self._p2.log_likelihood(samples)

        return lower_bound

