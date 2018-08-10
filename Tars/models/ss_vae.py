from copy import copy

import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist, get_dict_values


class SemiVAE(Model):
    def __init__(self, encoder, decoder,
                 ss_encoder, ss_decoder,
                 ss_rate=1,
                 other_distributions=[],
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(SemiVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.ss_encoder = ss_encoder
        self.ss_decoder = ss_decoder
        self.ss_rate = ss_rate
        self.regularizer = tolist(regularizer)

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

    def train(self, train_x, ss_train_x, coef=1):
        self.decoder.train()
        self.encoder.train()
        for distribution in self.other_distributions:
            distribution.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, ss_train_x, coef)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x, ss_test_x, coef=1):
        self.decoder.eval()
        self.encoder.eval()
        for distribution in self.other_distributions:
            distribution.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x, ss_test_x, coef)

        return lower_bound, loss

    def _elbo(self, x, ss_x, reg_coef=[1]):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """
        if not set(list(x.keys())) == set(self.input_var):
            raise ValueError("Input's keys are not valid.")
        reg_coef = tolist(reg_coef)

        # reconstruction error
        _x = get_dict_values(x, self.encoder.cond_var, True)
        samples = self.encoder.sample(_x)
        log_like = self.decoder.log_likelihood(samples)

        # reconstruction error (semi-supervised)
        _x = get_dict_values(ss_x, self.ss_encoder.cond_var, True)
        samples = self.ss_encoder.sample(_x)
        ss_log_like = self.ss_decoder.log_likelihood(samples)

        # regularization term
        lower_bound = [log_like]
        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate(x)
            lower_bound.append(_reg)
            reg_loss += reg_coef[i] * _reg

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(log_like - reg_loss) -\
            self.ss_rate * torch.mean(ss_log_like)

        return lower_bound, loss
