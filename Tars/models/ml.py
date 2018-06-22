import torch

from ..models.model import Model


class ML(Model):
    def __init__(self, p,
                 optimizer, optimizer_params={}):
        super(ML, self).__init__()

        self.p = p

        # set params and optim
        self.optimizer = optimizer(self.p.parameters(),
                                   **optimizer_params)

    def train(self, train_x):
        self.p.train()

        self.optimizer.zero_grad()
        log_like = self.p.log_likelihood(train_x)
        loss = -torch.mean(log_like)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return log_like, loss

    def test(self, test_x):
        self.p.eval()

        with torch.no_grad():
            log_like = self.p.log_likelihood(test_x)
            loss = -torch.mean(log_like)

        return log_like, loss
