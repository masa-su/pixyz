from .losses import Loss


class NLL(Loss):
    def __init__(self, p, input_var=[]):
        super().__init__(p, input_var=input_var)
        self.p = p
        self.loss_text = "log {}".format(self.p.prob_text)

    def estimate(self, x, **kwargs):
        _x = super().estimate(x)
        nll = -self.p.log_likelihood(x)

        return nll
