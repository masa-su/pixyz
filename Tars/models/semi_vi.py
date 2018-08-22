from copy import copy

import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist, get_dict_values


class SemiVI(Model):
    def __init__(self, p, approximate_dist,
                 discriminator,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(SemiVI, self).__init__()

        self.p = p
        self.q = approximate_dist
        self.d = discriminator
        self.regularizer = tolist(regularizer)

        self.input_var = copy(self.q.cond_var)

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        d_params = list(self.d.parameters())
        params = q_params + p_params + d_params

        self.other_distributions = tolist(other_distributions)
        for distribution in other_distributions:
            params += list(distribution.parameters())
            self.input_var += distribution.cond_var

        self.input_var = list(set(self.input_var))
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, u_train_x, supervised_rate=1, **kwargs):
        self.p.train()
        self.q.train()
        self.d.train()
        for distribution in self.other_distributions:
            distribution.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, **kwargs)
        _, u_loss = self._u_elbo(u_train_x, **kwargs)
        d_loss = self._d_loss(train_x)

        loss = loss + u_loss + supervised_rate * d_loss

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x, supervised_rate=1, **kwargs):
        self.p.eval()
        self.q.eval()
        for distribution in self.other_distributions:
            distribution.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x, **kwargs)

        return lower_bound, loss

    def _elbo(self, x, reg_coef=[], **kwargs):
        """
        The evidence lower bound
        """

        lower_bound = []
        # lower bound
        samples = self.q.sample(x, **kwargs)
        lower_bound.append(self.p.log_likelihood(samples) -
                           self.q.log_likelihood(samples))

        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate(x)
            lower_bound.append(_reg)
            reg_loss += reg_coef[i] * _reg

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(lower_bound)

        return lower_bound, loss

    def _u_elbo(self, x, reg_coef=[], **kwargs):
        disc_x = get_dict_values(x, self.d.cond_var, True)
        pred_x = self.d.sample(disc_x, return_all=False)
        x.update(pred_x)

        return self._elbo(x, reg_coef, **kwargs)

    def _d_loss(self, x):
        log_like = self.d.log_likelihood(x)
        loss = -torch.mean(log_like)

        return loss

