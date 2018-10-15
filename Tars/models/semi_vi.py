from copy import copy

import torch
from torch import optim, nn

from ..models.model import Model
from ..utils import tolist, get_dict_values


class SemiVI(Model):
    def __init__(self, p, approximate_dist,
                 discriminator,
                 other_distributions=[],
                 other_losses=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(SemiVI, self).__init__()

        self.p = p
        self.q = approximate_dist
        self.d = discriminator
        self.other_distributions = nn.ModuleList(tolist(other_distributions))

        self.other_losses = other_losses

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        d_params = list(self.d.parameters())
        other_params = list(self.other_distributions.parameters())
        params = q_params + p_params + d_params + other_params
        self.optimizer = optimizer(params, **optimizer_params)

        # set input_var
        self.input_var = copy(self.q.cond_var)
        for distribution in self.other_distributions:
            self.input_var += copy(distribution.cond_var)
        self.input_var = list(set(self.input_var))

    def train(self, train_x, u_train_x, supervised_rate=1, **kwargs):
        self.p.train()
        self.q.train()
        self.d.train()
        self.other_distributions.train()

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

    def test(self, test_x, **kwargs):
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

    def _u_elbo(self, x, **kwargs):
        disc_x = get_dict_values(x, self.d.cond_var, True)
        pred_x = self.d.sample(disc_x, return_all=False)
        x.update(pred_x)

        return self._elbo(x, **kwargs)

    def _d_loss(self, x):
        log_like = self.d.log_likelihood(x)
        loss = -torch.mean(log_like)

        return loss
