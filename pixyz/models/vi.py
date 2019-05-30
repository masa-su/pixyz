from torch import optim

from ..models.model import Model
from ..utils import tolist
from ..losses import ELBO


class VI(Model):
    """
    Variational Inference (Amortized inference)

    The ELBO over given distributions (p, approximate_dist) is set as the loss class of this model.

    """
    def __init__(self, p, approximate_dist,
                 other_distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):

        # set distributions (for training)
        distributions = [p, approximate_dist] + tolist(other_distributions)

        # set losses
        elbo = ELBO(p, approximate_dist)
        loss = -elbo.mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)
