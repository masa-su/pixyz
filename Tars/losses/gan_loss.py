from torch import optim
import torch
from .losses import Loss
from ..utils import get_dict_values


class GANLoss(Loss):
    """
    GAN loss.
    This Loss class has an optimizer for discriminator.
    """
    def __init__(self, p_data, p, discriminator, input_var=[], optimizer=optim.Adam, optimizer_params={}):
        super().__init__(p_data, p, input_var=input_var)
        self.loss_optimizer = optimizer
        self.loss_optimizer_params = optimizer_params
        self.d = discriminator

        params = discriminator.parameters()
        self.d_optimizer = optimizer(params, **optimizer_params)

    @property
    def loss_text(self):
        return "GANLoss[{}||{}]".format(self._p1.prob_text,
                                        self._p2.prob_text)

    def estimate(self, x={}, discriminator=False):
        _x = super().estimate(x)
        x_data = get_dict_values(_x, self._p1.input_var)[0]
        batch_size = x_data.shape[0]

        sample_dict = (self.d * self._p2).sample(batch_size=batch_size)
        sample = get_dict_values(sample_dict, self.d.var)[0]

        if discriminator:
            x_data_dict = get_dict_values(_x, self._p1.input_var, True)
            sample_data_dict = self.d.sample(x_data_dict)
            sample_data = get_dict_values(sample_data_dict, self.d.var)[0]
            return self.d_criterion(sample_data, sample)  # TODO: detach

        return self.g_criterion(sample)

    @staticmethod
    def d_criterion(sample_data, sample):
        return - torch.log(sample_data) - torch.log(1 - sample)

    @staticmethod
    def g_criterion(sample):
        return - torch.log(sample)

    def train(self, train_x, **kwargs):
        self.d.train()

        self.d_optimizer.zero_grad()
        loss = torch.mean(self.estimate(train_x, discriminator=True), dim=0)

        # backprop
        loss.backward()

        # update params
        self.d_optimizer.step()

        return loss

    def test(self, test_x, **kwargs):
        self.d.eval()

        with torch.no_grad():
            loss = torch.mean(self.estimate(test_x, **kwargs))

        return loss
