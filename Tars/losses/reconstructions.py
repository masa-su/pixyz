from ..utils import get_dict_values
from .losses import Loss


class StochasticReconstructionLoss(Loss):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def estimate(self, x, **kwargs):
        _x = get_dict_values(x, self.encoder.cond_var, True)
        samples = self.encoder.sample(_x)
        loss = -self.decoder.log_likelihood(samples)

        return loss
