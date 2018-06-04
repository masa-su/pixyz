import numpy as np
import torch

from ..utils import get_dict_values
from ..distributions.estimate_kl import analytical_kl
from ..models.model import Model

class VAE(Model):
    def __init__(self, q, p, prior,
                 optimizer, optimizer_params={},
                 seed=1234):
        super(VAE, self).__init__()

        self.q = q
        self.p = p
        self.prior = prior

        # set params and optim
        q_params = list(self.q.parameters())
        p_params = list(self.p.parameters())
        params = q_params + p_params
        
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, annealing_beta=1):
        self.p.train()
        self.q.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, annealing_beta)

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

    def _elbo(self, x, annealing_beta=1):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """
        samples = self.q.sample(x)
        log_like = self.p.log_likelihood(samples)

        kl = analytical_kl(self.q, self.prior, given=[x, None])

        lower_bound = torch.stack((-kl, log_like), dim=-1)
        loss = -torch.mean(log_like - annealing_beta * kl)

        return lower_bound, loss
