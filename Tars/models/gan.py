from torch import optim, nn

from ..models.model import Model
from ..losses import AdversarialJSDivergence


class GAN(Model):
    """
    Generative adversarial Network
    """
    def __init__(self, p_data, p, discriminator,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 d_optimizer=optim.Adam,
                 d_optimizer_params={},):

        distributions = nn.ModuleList([p])
        super().__init__(distributions)

        # set losses
        loss_cls = AdversarialJSDivergence(p_data, p, discriminator,
                                           optimizer=d_optimizer, optimizer_params=d_optimizer_params).mean()
        self.loss_cls = loss_cls
        self.test_loss_cls = loss_cls
        self.loss_text = str(loss_cls)

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

    def train(self, train_x, adversarial_loss=True, **kwargs):
        if adversarial_loss:
            d_loss = self.loss_cls.train(train_x, **kwargs)
        loss = super().train(train_x, **kwargs)

        if adversarial_loss:
            return loss, d_loss

        return loss

    def test(self, test_x, adversarial_loss=True, **kwargs):
        loss = super().test(test_x, **kwargs)
        if adversarial_loss:
            d_loss = self.loss_cls.test(test_x, **kwargs)
            return loss, d_loss
        return loss

