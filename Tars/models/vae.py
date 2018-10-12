from copy import copy

import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist
from ..losses import StochasticReconstructionLoss


class VAE(Model):
    def __init__(self, encoder, decoder,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.regularizer = regularizer

        self.reconstruction =\
            StochasticReconstructionLoss(self.encoder, self.decoder)

        self.input_var = copy(self.encoder.cond_var)

        # set params and optim
        q_params = list(self.encoder.parameters())
        p_params = list(self.decoder.parameters())
        params = q_params + p_params

        self.other_distributions = tolist(other_distributions)
        for distribution in other_distributions:
            params += list(distribution.parameters())
            self.input_var += distribution.cond_var

        self.input_var = list(set(self.input_var))
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x):
        self.decoder.train()
        self.encoder.train()
        for distribution in self.other_distributions:
            distribution.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x):
        self.decoder.eval()
        self.encoder.eval()
        for distribution in self.other_distributions:
            distribution.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x)

        return lower_bound, loss

    def _elbo(self, x):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """
        if not set(list(x.keys())) == set(self.input_var):
            raise ValueError("Input's keys are not valid.")

        # negative reconstruction error
        log_like = -self.reconstruction.estimate(x)
        lower_bound = [log_like]

        # regularization term
        reg_loss = self.regularizer.estimate(x)
        lower_bound.append(reg_loss)

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(log_like - reg_loss)

        return lower_bound, loss
