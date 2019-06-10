from torch import optim

from ..models.model import Model
from ..utils import tolist
from ..losses import StochasticReconstructionLoss


class VAE(Model):
    """
    Variational Autoencoder.

    In VAE class, reconstruction loss on given distributions (encoder and decoder) is set as the default loss class.
    However, if you want to add additional terms, e.g., the KL divergence between encoder and prior,
    you need to set them to the `regularizer` argument, which defaults to None.

    References
    ----------
    [Kingma+ 2013] Auto-Encoding Variational Bayes
    """
    def __init__(self, encoder, decoder,
                 other_distributions=[],
                 regularizer=None,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None):
        """
        Parameters
        ----------
        encoder : torch.distributions.Distribution
            Encoder distribution.
        decoder : torch.distributions.Distribution
            Decoder distribution.
        regularizer : torch.losses.Loss, defaults to None
            If you want to add additional terms to the loss, set them to this argument.
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
        distributions = [encoder, decoder] + tolist(other_distributions)

        # set losses
        reconstruction =\
            StochasticReconstructionLoss(encoder, decoder)
        loss = (reconstruction + regularizer).mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)
