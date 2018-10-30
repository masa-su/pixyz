from torch import optim, nn
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

        self.bce_loss = nn.BCELoss()

    @property
    def loss_text(self):
        return "GANLoss[{}||{}]".format(self._p1.prob_text,
                                        self._p2.prob_text)

    def estimate(self, x={}, discriminator=False):
        _x = super().estimate(x)

        # sample x from p1 (p_data)
        x_data_dict = get_dict_values(_x, self._p1.input_var, True)
        # x_data = self._p1.sample(x_data)
        x_data = x_data_dict[self._p1.var[0]]
        batch_size = x_data.shape[0]

        # sample x from p2 (p)
        x_dict = self._p2.sample(batch_size=batch_size)

        # set labels
        t_data = torch.ones(batch_size, 1).to(x_data.device)
        t = torch.zeros(batch_size, 1).to(x_data.device)

        if discriminator:
            # sample y from x_data
            y_data_dict = self.d.sample(x_data_dict)
            y_data = get_dict_values(y_data_dict, self.d.var)[0]

            # sample y from x
            y_dict = self.d.sample(x_dict)  # TODO: detach x_dict
            y = get_dict_values(y_dict, self.d.var)[0]

            return self.bce_loss(y_data, t_data) + self.bce_loss(y, t)

        # sample y from x
        y_dict = self.d.sample(x_dict)
        y = get_dict_values(y_dict, self.d.var)[0]

        return self.bce_loss(y, t_data)

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
