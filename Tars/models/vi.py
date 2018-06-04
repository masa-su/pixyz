import numpy as np
import torch

from ..utils import get_dict_values
from ..distributions.estimate_kl import analytical_kl
from ..models.model import Model

class VI(Model):
    def __init__(self, p, approximate_dist,
                 optimizer, optimizer_params={},
                 seed=1234):
        super(VI, self).__init__()

        self.p = p
        self.q = approximate_dist

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        params = q_params + p_params
        
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, annealing_beta=1):
        self.p.train()
        self.q.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x):
        self.p.eval()
        self.q.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x)

        return lower_bound, loss

    def _elbo(self, x):
        """
        The evidence lower bound
        """
        samples = self.q.sample(x)
        lower_bound = self.p.log_likelihood(samples) -\
                      self.q.log_likelihood(samples)

        loss = -torch.mean(lower_bound)

        return lower_bound, loss
