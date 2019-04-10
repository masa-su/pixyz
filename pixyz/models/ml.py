from torch import optim, nn

from ..models.model import Model
from ..utils import tolist


class ML(Model):
    """
    Maximum Likelihood (log-likelihood)
    """
    def __init__(self, p,
                 other_distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):

        # set distributions (for training)
        distributions = [p] + tolist(other_distributions)

        # set losses
        self.nll = -p.log_prob(sum_features=True)
        loss = self.nll.mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params)

    def train(self, train_x={}, **kwargs):
        return super().train(train_x, **kwargs)

    def test(self, test_x={}, **kwargs):
        return super().test(test_x, **kwargs)

