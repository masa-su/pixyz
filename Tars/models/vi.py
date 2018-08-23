from copy import copy

import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist


class VI(Model):
    def __init__(self, p, approximate_dist,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(VI, self).__init__()

        self.p = p
        self.q = approximate_dist
        self.regularizer = tolist(regularizer)

        self.input_var = copy(self.q.cond_var)

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        params = q_params + p_params

        self.other_distributions = tolist(other_distributions)
        for distribution in other_distributions:
            params += list(distribution.parameters())
            self.input_var += distribution.cond_var

        self.input_var = list(set(self.input_var))
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x=None, coef=[], **kwargs):
        self.p.train()
        self.q.train()
        for distribution in self.other_distributions:
            distribution.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, coef, **kwargs)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x=None, coef=[], **kwargs):
        self.p.eval()
        self.q.eval()
        for distribution in self.other_distributions:
            distribution.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x, coef, **kwargs)

        return lower_bound, loss

    def _elbo(self, x, reg_coef=[], **kwargs):
        """
        The evidence lower bound
        """
        if not set(list(x.keys())) == set(self.input_var):
            raise ValueError("Input's keys are not valid.")
        reg_coef = tolist(reg_coef)

        lower_bound = []
        # lower bound
        samples = self.q.sample(x, **kwargs)
        _lower_bound = self.p.log_likelihood(samples) -\
            self.q.log_likelihood(samples)
        lower_bound.append(_lower_bound)

        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate(x)
            lower_bound.append(_reg)
            reg_loss += reg_coef[i] * _reg

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(_lower_bound - reg_loss)

        return lower_bound, loss
