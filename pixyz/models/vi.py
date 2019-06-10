from torch import optim

from ..models.model import Model
from ..utils import tolist
from ..losses import ELBO


class VI(Model):
    """
    Variational Inference (Amortized inference)

    The ELBO for given distributions (p, approximate_dist) is set as the loss class of this model.

    """
    def __init__(self, p, approximate_dist,
                 other_distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None):
        """
        Parameters
        ----------
        p : torch.distributions.Distribution
            Generative model (distribution).
        approximate_dist : torch.distributions.Distribution
            Approximate posterior distribution.
        optimizer : torch.optim
            Optimization algorithm.
        optimizer_params : dict
            Parameters of optimizer
        clip_grad_norm : float or int
            Maximum allowed norm of the gradients.
        clip_grad_value : float or int
            Maximum allowed value of the gradients.
        """

        # set distributions (for training)
        distributions = [p, approximate_dist] + tolist(other_distributions)

        # set losses
        elbo = ELBO(p, approximate_dist)
        loss = -elbo.mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)
