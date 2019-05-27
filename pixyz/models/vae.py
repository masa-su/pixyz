from torch import optim

from ..models.model import Model
from ..utils import tolist
from ..losses import StochasticReconstructionLoss


class VAE(Model):
    """
    Variational Autoencoder

    [Kingma+ 2013] Auto-Encoding Variational Bayes
    """
    def __init__(self, encoder, decoder,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):

        # set distributions (for training)
        distributions = [encoder, decoder] + tolist(other_distributions)

        # set losses
        reconstruction =\
            StochasticReconstructionLoss(encoder, decoder)
        loss = (reconstruction + regularizer).mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)
