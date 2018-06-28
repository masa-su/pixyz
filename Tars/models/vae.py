import torch
from torch import optim

from ..models.model import Model
from ..utils import tolist


class VAE(Model):
    def __init__(self, encoder, decoder,
                 regularizer=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.regularizer = tolist(regularizer)

        # set params and optim
        q_params = list(self.encoder.parameters())
        p_params = list(self.decoder.parameters())
        params = q_params + p_params

        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, coef=1):
        self.decoder.train()
        self.encoder.train()

        self.optimizer.zero_grad()
        lower_bound, loss = self._elbo(train_x, coef)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return lower_bound, loss

    def test(self, test_x):
        self.decoder.eval()
        self.encoder.eval()

        with torch.no_grad():
            lower_bound, loss = self._elbo(test_x)

        return lower_bound, loss

    def _elbo(self, x, reg_coef=[1]):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """
        reg_coef = tolist(reg_coef)

        # reconstrunction error
        samples = self.encoder.sample(x)
        log_like = self.decoder.log_likelihood(samples)

        # regularization term
        lower_bound = [log_like]
        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate(x)
            lower_bound.append(_reg)
            reg_loss += reg_coef[i] * _reg

        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(log_like - reg_loss)

        return lower_bound, loss
