from .losses import Loss


class StochasticReconstructionLoss(Loss):
    def __init__(self, encoder, decoder, input_var=[]):
        if len(input_var) == 0:
            input_var = encoder.input_var
        super().__init__(encoder, decoder, input_var=input_var)

    @property
    def loss_text(self):
        return "E_{}[log {}]".format(self._p1.prob_text, self._p2.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        samples = self._p1.sample(_x, reparam=True)
        loss = -self._p2.log_likelihood(samples)

        return loss
