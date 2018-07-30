import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist


class Regularization(Model):
    def __init__(self, regularizer=[],
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(Regularization, self).__init__()

        self.regularizer = tolist(regularizer)
        self.distributions = distributions

        # set params and optim
        params = []
        distributions = tolist(distributions)
        for distribution in distributions:
            params += list(distribution.parameters())

        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, coef=[1]):
        for distribution in self.distributions:
            distribution.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, coef)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x, coef=1):
        for distribution in self.distributions:
            distribution.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x, coef)

        return lower_bound, loss

    def _elbo(self, x, reg_coef=[1]):
        reg_coef = tolist(reg_coef)

        # regularization term
        lower_bound = []
        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate(x)
            lower_bound.append(-_reg)
            reg_loss += reg_coef[i] * _reg

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = torch.mean(reg_loss)

        return lower_bound, loss
