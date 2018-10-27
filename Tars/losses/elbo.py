from .losses import Loss


class ELBO(Loss):
    """
    The evidence lower bound
    """
    def __init__(self, p, approximate_dist, input_var=[]):
        if len(input_var) == 0:
            input_var = approximate_dist.cond_var  # TODO: fix to input_var

        super().__init__(p, approximate_dist, input_var=input_var)

    @property
    def loss_text(self):
        return "E_{}[log {}/{}]".format(self._p2.prob_text,
                                        self._p1.prob_text,
                                        self._p2.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        samples = self._p2.sample(_x, reparam=True)
        lower_bound = self._p1.log_likelihood(samples) -\
            self._p2.log_likelihood(samples)

        return lower_bound

