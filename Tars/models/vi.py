from torch import optim, nn

from ..models.model import Model
from ..utils import tolist
from ..losses import ELBO


class VI(Model):
    def __init__(self, p, approximate_dist,
                 other_distributions=[],
                 other_losses=None,
                 optimizer=optim.Adam,
                 optimizer_params={}):

        distributions = nn.ModuleList([p, approximate_dist] + tolist(other_distributions))
        super().__init__(distributions)

        # set losses
        elbo = ELBO(p, approximate_dist)
        other_losses = other_losses
        loss_cls = (-elbo + other_losses).mean()
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
