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
        super(CustomLossModel, self).__init__()

        self.distributions = nn.ModuleList(tolist(distributions))

        self.loss_cls = None
        self.test_loss_cls = None
        self.input_var = None

        self.set_loss(loss, test_loss)

        # set params and optim
        params = list(self.distributions.parameters())
        self.optimizer = optimizer(params, **optimizer_params)

    def set_loss(self, loss, test_loss=None):
        self.loss_cls = loss
        self.input_var = self.loss_cls.input_var
        if test_loss:
            self.test_loss_cls = test_loss
        else:
            self.test_loss_cls = loss

    def train(self, train_x):
        self.distributions.train()

        self.optimizer.zero_grad()
        loss = self.loss_cls.estimate(train_x).mean()

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return loss

    def test(self, test_x):
        self.distributions.eval()

        with torch.no_grad():
            loss = self.test_loss_cls.estimate(test_x).mean()

        return loss
