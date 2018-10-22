from .losses import Loss


class StochasticReconstructionLoss(Loss):
    def __init__(self, encoder, decoder, input_var=[]):
        super(StochasticReconstructionLoss, self).__init__(encoder, input_var=input_var)
        self.encoder = encoder
        self.decoder = decoder

        self.loss_text = "E_{}[log {}]".format(self.encoder.prob_text,
                                               self.decoder.prob_text)

    def estimate(self, x, **kwargs):
        _x = super(StochasticReconstructionLoss, self).estimate(x)
        samples = self.encoder.sample(_x)
        loss = -self.decoder.log_likelihood(samples)

        return loss
