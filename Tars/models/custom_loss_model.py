import torch
from torch import optim, nn

from ..models.model import Model
from ..utils import tolist


class CustomLossModel(Model):
    def __init__(self, loss=[],
                 test_loss=None,
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super().__init__()

        self.distributions = nn.ModuleList(tolist(distributions))

        # set losses
        self.loss_cls = None
        self.test_loss_cls = None
        self.set_loss(loss, test_loss)

        # set params and optim
        params = list(self.distributions.parameters())
        self.optimizer = optimizer(params, **optimizer_params)

    def set_loss(self, loss, test_loss=None):
        self.loss_cls = loss
        if test_loss:
            self.test_loss_cls = test_loss
        else:
            self.test_loss_cls = loss

    def train(self, train_x, **kwargs):
        self.distributions.train()

        return super().train(train_x, **kwargs)

    def test(self, test_x, **kwargs):
        self.distributions.eval()

        return super().teset(test_x, **kwargs)

