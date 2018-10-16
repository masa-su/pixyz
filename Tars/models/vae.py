from copy import copy

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
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.other_distributions = nn.ModuleList(tolist(other_distributions))

        # set losses
        reconstruction =\
            StochasticReconstructionLoss(self.encoder, self.decoder)
        loss_cls = (reconstruction + regularizer).mean()
        self.loss_cls = loss_cls
        self.test_loss_cls = loss_cls
        self.loss_text = str(loss_cls)

        # set params and optim
        q_params = list(self.encoder.parameters())
        p_params = list(self.decoder.parameters())
        other_params = list(self.other_distributions.parameters())
        params = q_params + p_params + other_params
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, **kwargs):
        self.decoder.train()
        self.encoder.train()
        self.other_distributions.train()

        return super().train(train_x, **kwargs)

    def test(self, test_x, **kwargs):
        self.decoder.eval()
        self.encoder.eval()
        self.other_distributions.eval()

        return super().test(test_x, **kwargs)
