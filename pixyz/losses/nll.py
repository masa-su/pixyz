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
        return "-log {}".format(self._p1.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        nll = -self._p1.log_likelihood(x)

        return nll
