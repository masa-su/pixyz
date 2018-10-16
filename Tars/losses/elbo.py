from .losses import Loss


class ELBO(Loss):
    def __init__(self, p, approximate_dist, input_var=[]):
        super(ELBO, self).__init__(approximate_dist,
                                   input_var=input_var)
        self.p = p
        self.q = approximate_dist

    def estimate(self, x, **kwargs):
        _x = super(ELBO, self).estimate(x)
        samples = self.q.sample(x, **kwargs)
        lower_bound = self.p.log_likelihood(samples) -\
            self.q.log_likelihood(samples)

        return lower_bound