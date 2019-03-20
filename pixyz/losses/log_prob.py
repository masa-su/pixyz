from .losses import Loss


class LogProb(Loss):
    r"""
    The log probability density/mass function.

    .. math::

        \log p(x)
    """

    def __init__(self, p):
        input_var = p.var + p.cond_var
        super().__init__(p, input_var=input_var)

    @property
    def loss_text(self):
        return "log {}".format(self._p1.prob_text)

    def _get_estimated_value(self, x={}, **kwargs):
        log_prob = -self._p1.log_likelihood(x)
        return log_prob, x
