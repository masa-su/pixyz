from torch import optim, nn

from ..models.model import Model
from ..utils import tolist
from ..losses import StochasticReconstructionLoss


class VAE(Model):
    """
    The evidence lower bound (original VAE)
    [Kingma+ 2013] Auto-Encoding Variational Bayes
    """
    def __init__(self, encoder, decoder,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):

        distributions = nn.ModuleList([encoder, decoder] + tolist(other_distributions))
        super().__init__(distributions)

        # set losses
        reconstruction =\
            StochasticReconstructionLoss(encoder, decoder)
        loss_cls = (reconstruction + regularizer).mean()
        self.loss_cls = loss_cls
        self.test_loss_cls = loss_cls
        self.loss_text = str(loss_cls)

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, **kwargs):
        return super().train(train_x, **kwargs)

    def test(self, test_x, **kwargs):
        return super().test(test_x, **kwargs)
