from copy import copy

import torch
from torch import optim, nn

from ..models.model import Model
from ..utils import tolist


class VI(Model):
    def __init__(self, p, approximate_dist,
                 other_distributions=[],
                 other_losses=None,
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(VI, self).__init__()

        self.p = p
        self.q = approximate_dist
        self.other_distributions = nn.ModuleList(tolist(other_distributions))

        self.other_losses = other_losses

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        other_params = list(self.other_distributions.parameters())
        params = q_params + p_params + other_params
        self.optimizer = optimizer(params, **optimizer_params)

        # set input_var
        self.input_var = copy(self.q.cond_var)
        for distribution in self.other_distributions:
            self.input_var += copy(distribution.cond_var)
        self.input_var = list(set(self.input_var))

    def train(self, train_x=None, **kwargs):
        self.p.train()
        self.q.train()
        self.other_distributions.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, **kwargs)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x=None, **kwargs):
        self.p.eval()
        self.q.eval()
        self.other_distributions.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x, **kwargs)

        return lower_bound, loss

    def _elbo(self, x, **kwargs):
        """
        The evidence lower bound
        """
        if not set(list(x.keys())) == set(self.input_var):
            raise ValueError("Input's keys are not valid.")

        lower_bound = []
        # lower bound
        samples = self.q.sample(x, **kwargs)
        _lower_bound = self.p.log_likelihood(samples) -\
            self.q.log_likelihood(samples)
        lower_bound.append(_lower_bound)

        # other losses
        loss = self.other_losses.estimate(x)
        lower_bound.append(loss)

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(_lower_bound - loss)

        return lower_bound, loss
