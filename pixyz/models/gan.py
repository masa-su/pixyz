from torch import optim

from ..models.model import Model
from ..losses import AdversarialJensenShannon
from ..distributions import DataDistribution


class GAN(Model):
    """
    Generative Adversarial Network

    (Adversarial) Jensen-Shannon divergence between given distributions (p_data, p)
    is set as the loss class of this model.
    """
    def __init__(self, p, discriminator,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 d_optimizer=optim.Adam,
                 d_optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None):
        """
        Parameters
        ----------
        p : torch.distributions.Distribution
            Generative model (generator).
        discriminator : torch.distributions.Distribution
            Critic (discriminator).
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
        distributions = [p]
        p_data = DataDistribution(p.var)

        # set losses
        loss = AdversarialJensenShannon(p_data, p, discriminator,
                                        optimizer=d_optimizer, optimizer_params=d_optimizer_params)

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

    def train(self, train_x_dict={}, adversarial_loss=True, **kwargs):
        """Train the model.

        Parameters
        ----------
        train_x_dict : dict, defaults to {}
            Input data.
        adversarial_loss : bool, defaults to True
            Whether to train the discriminator.
        **kwargs

        Returns
        -------
        loss : torch.Tensor
            Train loss value.

        d_loss : torch.Tensor
            Train loss value of the discriminator (if :attr:`adversarial_loss` is True).

        """
        if adversarial_loss:
            d_loss = self.loss_cls.train(train_x_dict, **kwargs)
        loss = super().train(train_x_dict, **kwargs)

        if adversarial_loss:
            return loss, d_loss

        return loss

    def test(self, test_x_dict={}, adversarial_loss=True, **kwargs):
        """Train the model.

        Parameters
        ----------
        test_x_dict : dict, defaults to {}
            Input data.
        adversarial_loss : bool, defaults to True
            Whether to return the discriminator loss.
        **kwargs

        Returns
        -------
        loss : torch.Tensor
            Test loss value.

        d_loss : torch.Tensor
            Test loss value of the discriminator (if :attr:`adversarial_loss` is True).

        """
        loss = super().test(test_x_dict, **kwargs)
        if adversarial_loss:
            d_loss = self.loss_cls.test(test_x_dict, **kwargs)
            return loss, d_loss
        return loss
