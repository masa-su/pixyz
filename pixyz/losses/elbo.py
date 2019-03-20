from .losses import Loss


class ELBO(Loss):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),

    where :math:`z_l \sim q(z|x)`.
    """
    def __init__(self, p, q, input_var=None):
        if input_var is None:
            input_var = q.input_var

        super().__init__(p, q, input_var=input_var)

    @property
    def loss_text(self):
        return "E_{}[log {}/{}]".format(self._q.prob_text,
                                        self._p.prob_text,
                                        self._q.prob_text)

    def _get_estimated_value(self, x={}, batch_size=None, **kwargs):
        samples_dict = self._q.sample(x, reparam=True, batch_size=batch_size)

        lower_bound = self._p.log_likelihood(samples_dict) - self._q.log_likelihood(samples_dict)

        return lower_bound, samples_dict

