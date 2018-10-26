from .losses import Loss


class ELBO(Loss):
    """
    The evidence lower bound
    """
    def __init__(self, p, approximate_dist, input_var=[]):
        super().__init__(approximate_dist, input_var=input_var)
        self.p = p
        self.q = approximate_dist
        self.loss_text = "E_{}[log {}/{}]".format(self.q.prob_text,
                                                  self.p.prob_text,
                                                  self.q.prob_text)

    def estimate(self, x, **kwargs):
        _x = super().estimate(x)
        samples = self.q.sample(_x, reparam=True)
        lower_bound = self.p.log_likelihood(samples) -\
            self.q.log_likelihood(samples)

        return lower_bound

