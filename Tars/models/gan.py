from torch import optim, nn

from ..models.model import Model
from ..losses import GANLoss


class GAN(Model):
    """
    Generative adversarial Network
    """
    def __init__(self, p_data, p, discriminator,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 d_optimizer=optim.Adam,
                 d_optimizer_params = {},):

        distributions = nn.ModuleList([p])
        super().__init__(distributions)

        # set losses
        loss_cls = GANLoss(p_data, p, discriminator,
                           optimizer=d_optimizer, optimizer_params=d_optimizer_params).mean()
        self.loss_cls = loss_cls
        self.test_loss_cls = loss_cls
        self.loss_text = str(loss_cls)

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, **kwargs):
        return super().train(train_x, adversarial_loss=True, **kwargs)

    def test(self, test_x, **kwargs):
        return super().test(test_x, adversarial_loss=True, **kwargs)
