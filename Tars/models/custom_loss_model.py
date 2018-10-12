import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist


class CustomLossModel(Model):
    def __init__(self, loss=[],
                 test_loss=None,
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(CustomLossModel, self).__init__()

        self.loss_cls = None
        self.test_loss_cls = None

        self.set_loss(loss, test_loss)
        self.distributions = distributions

        # set params and optim
        params = []
        distributions = tolist(distributions)
        for distribution in distributions:
            params += list(distribution.parameters())

        self.optimizer = optimizer(params, **optimizer_params)

    def set_loss(self, loss, test_loss=None):
        self.loss_cls = loss
        if test_loss:
            self.test_loss_cls = test_loss
        else:
            self.test_loss_cls = loss

    def train(self, train_x):
        for distribution in self.distributions:
            distribution.train()

        self.optimizer.zero_grad()
        loss = self.loss_cls.estimate(train_x).mean()

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return loss

    def test(self, test_x):
        for distribution in self.distributions:
            distribution.eval()

        with torch.no_grad():
            loss = self.test_loss_cls.estimate(test_x).mean()

        return loss
