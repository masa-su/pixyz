from torch import optim

from ..models.model import Model
from ..utils import tolist


class ML(Model):
    """
    Maximum Likelihood (log-likelihood)

    The negative log-likelihood of a given distribution (p) is set as the loss class of this model.
    """
    def __init__(self, p,
                 other_distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=False,
                 clip_grad_value=False):
        """
        Parameters
        ----------
        p : torch.distributions.Distribution
            Classifier (distribution).
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
        distributions = [p] + tolist(other_distributions)

        # set losses
        self.nll = -p.log_prob(sum_features=True)
        loss = self.nll.mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)
