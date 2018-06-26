import torch
from torch import optim

from ..models.model import Model


class VAE(Model):
    def __init__(self, encoder, decoder,
                 regularizer,
                 additional_regularizer=None,
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.regularizer = regularizer
        self.additional_regularizer = additional_regularizer

        # set params and optim
        q_params = list(self.encoder.parameters())
        p_params = list(self.decoder.parameters())
        params = q_params + p_params

        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, annealing_beta=1):
        self.decoder.train()
        self.encoder.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, annealing_beta)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x):
        self.p.eval()
        self.encoder.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x)

        return lower_bound, loss

    def _elbo(self, x, annealing_beta=1):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """
        samples = self.encoder.sample(x)
        log_like = self.decoder.log_likelihood(samples)

        reg = self.regularizer.estimate(x)
        if self.additional_regularizer:
            reg += self.additional_regularizer.estimate(x)

        lower_bound = torch.stack((-reg, log_like), dim=-1)
        loss = -torch.mean(log_like - annealing_beta * reg)

        return lower_bound, loss
