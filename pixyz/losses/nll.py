from .losses import Loss


class NLL(Loss):
    r"""
    Negative log-likelihood.

    .. math::

        -\log p(x)
    """

    def __init__(self, p, input_var=None):
        super().__init__(p, input_var=input_var)

    @property
    def loss_text(self):
        return "-log {}".format(self._p.prob_text)

    def _get_estimated_value(self, x={}, **kwargs):
        nll = -self._p.log_likelihood(x)
        return nll, x

