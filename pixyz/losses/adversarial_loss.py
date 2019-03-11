from torch import optim, nn
import torch
from .losses import Loss
from ..utils import get_dict_values, detach_dict


class AdversarialLoss(Loss):
    def __init__(self, p1, p2, discriminator, input_var=None,
                 optimizer=optim.Adam, optimizer_params={}):
        super().__init__(p1, p2, input_var=input_var)
        self.loss_optimizer = optimizer
        self.loss_optimizer_params = optimizer_params
        self.d = discriminator

        params = discriminator.parameters()
        self.d_optimizer = optimizer(params, **optimizer_params)

        # TODO: fix this decision rule
        if p1.distribution_name == "Data distribution":
            self._p1_data_dist = True
        else:
            self._p1_data_dist = False

        if p2.distribution_name == "Data distribution":
            self._p2_data_dist = True
        else:
            self._p2_data_dist = False

    def d_loss(self, y1, y2, batch_size):
        raise NotImplementedError

    def g_loss(self, y1, y2, batch_size):
        raise NotImplementedError

    def train(self, train_x, **kwargs):
        self.d.train()

        self.d_optimizer.zero_grad()
        loss = self.estimate(train_x, discriminator=True)

        # backprop
        loss.backward()

        # update params
        self.d_optimizer.step()

        return loss

    def test(self, test_x, **kwargs):
        self.d.eval()

        with torch.no_grad():
            loss = self.estimate(test_x, discriminator=True)

        return loss


class AdversarialJensenShannon(AdversarialLoss):
    r"""
    Jensen-Shannon divergence (adversarial training).

    .. math::

        D_{JS}[p(x)||q(x)] \leq 2 \cdot D_{JS}[p(x)||q(x)] + 2 \log 2
         = \mathbb{E}_{p(x)}[\log d^*(x)] + \mathbb{E}_{q(x)}[\log (1-d^*(x))],

    where :math:`d^*(x) = \arg\max_{d} \mathbb{E}_{p(x)}[\log d(x)] + \mathbb{E}_{q(x)}[\log (1-d(x))]`.
    """

    def __init__(self, p, q, discriminator, input_var=None, optimizer=optim.Adam, optimizer_params={},
                 inverse_g_loss=True):
        super().__init__(p, q, discriminator,
                         input_var=input_var,
                         optimizer=optimizer, optimizer_params=optimizer_params)

        self.bce_loss = nn.BCELoss()
        self._inverse_g_loss = inverse_g_loss

    @property
    def loss_text(self):
        return "mean(AdversarialJS[{}||{}])".format(self._p1.prob_text,
                                                    self._p2.prob_text)

    def _get_estimated_value(self, x, discriminator=False, **kwargs):
        batch_size = get_dict_values(x, self._p1.input_var[0])[0].shape[0]

        # sample x from p1
        x_dict = get_dict_values(x, self._p1.input_var, True)
        if self._p1_data_dist:
            x1_dict = x_dict
        else:
            x1_dict = self._p1.sample(x_dict, batch_size=batch_size)
            x1_dict = get_dict_values(x1_dict, self.d.input_var, True)

        # sample x from p2
        x_dict = get_dict_values(x, self._p2.input_var, True)
        x2_dict = self._p2.sample(x_dict, batch_size=batch_size)
        x2_dict = get_dict_values(x2_dict, self.d.input_var, True)

        if discriminator:
            # sample y from x1
            y1_dict = self.d.sample(detach_dict(x1_dict))
            y1 = get_dict_values(y1_dict, self.d.var)[0]

            # sample y from x2
            y2_dict = self.d.sample(detach_dict(x2_dict))
            y2 = get_dict_values(y2_dict, self.d.var)[0]

            return self.d_loss(y1, y2, batch_size), x

        # sample y from x1
        y1_dict = self.d.sample(x1_dict)
        # sample y from x2
        y2_dict = self.d.sample(x2_dict)

        y1 = get_dict_values(y1_dict, self.d.var)[0]
        y2 = get_dict_values(y2_dict, self.d.var)[0]

        return self.g_loss(y1, y2, batch_size), x  # TODO: fix

    def d_loss(self, y1, y2, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y1.device)
        t2 = torch.zeros(batch_size, 1).to(y1.device)
        return self.bce_loss(y1, t1) + self.bce_loss(y2, t2)

    def g_loss(self, y1, y2, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y1.device)
        t2 = torch.zeros(batch_size, 1).to(y1.device)

        if self._inverse_g_loss:
            y1_loss = self.bce_loss(y1, t2)
            y2_loss = self.bce_loss(y2, t1)
        else:
            y1_loss = -self.bce_loss(y1, t1)
            y2_loss = -self.bce_loss(y2, t2)

        if self._p1_data_dist:
            y1_loss = y1_loss.detach()

        return y1_loss + y2_loss


class AdversarialKullbackLeibler(AdversarialLoss):
    r"""
    Kullback-Leibler divergence (adversarial training).

    .. math::

        D_{KL}[q(x)||p(x)] = \mathbb{E}_{q(x)}[\log \frac{q(x)}{p(x)}]
         = \mathbb{E}_{q(x)}[\log \frac{d^*(x)}{1-d^*(x)}],

    where :math:`d^*(x) = \arg\max_{d} \mathbb{E}_{p(x)}[\log d(x)] + \mathbb{E}_{q(x)}[\log (1-d(x))]`.

    Note that this divergence is minimized to close q to p.
    """

    def __init__(self, q, p, discriminator, **kwargs):
        super().__init__(q, p, discriminator, **kwargs)
        self.bce_loss = nn.BCELoss()

    @property
    def loss_text(self):
        return "mean(AdversarialKL[{}||{}])".format(self._p1.prob_text,
                                                    self._p2.prob_text)

    def _get_estimated_value(self, x, discriminator=False, **kwargs):
        batch_size = get_dict_values(_x, self._p1.input_var[0])[0].shape[0]

        # sample x from p1
        x_dict = get_dict_values(_x, self._p1.input_var, True)
        x1_dict = self._p1.sample(x_dict, batch_size=batch_size)
        x1_dict = get_dict_values(x1_dict, self.d.input_var, True)

        if discriminator:
            # sample x from p2
            x_dict = get_dict_values(_x, self._p2.input_var, True)
            x2_dict = self._p2.sample(x_dict, batch_size=batch_size)
            x2_dict = get_dict_values(x2_dict, self.d.input_var, True)

            # sample y from x1
            y1_dict = self.d.sample(detach_dict(x1_dict))
            y1 = get_dict_values(y1_dict, self.d.var)[0]

            # sample y from x2
            y2_dict = self.d.sample(detach_dict(x2_dict))
            y2 = get_dict_values(y2_dict, self.d.var)[0]

            return self.d_loss(y1, y2, batch_size)

        # sample y from x1
        y1_dict = self.d.sample(x1_dict)
        y1 = get_dict_values(y1_dict, self.d.var)[0]

        return self.g_loss(y1, batch_size), x

    def g_loss(self, y1, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y1.device)
        t2 = torch.zeros(batch_size, 1).to(y1.device)

        y1_loss = -self.bce_loss(y1, t1) + self.bce_loss(y1, t2)

        return y1_loss

    def d_loss(self, y1, y2, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y1.device)
        t2 = torch.zeros(batch_size, 1).to(y1.device)
        return self.bce_loss(y1, t1) + self.bce_loss(y2, t2)


class AdversarialWassersteinDistance(AdversarialJensenShannon):
    r"""
    Wasserstein distance (adversarial training).

    .. math::

         W(p, q) = \sup_{||d||_{L} \leq 1} \mathbb{E}_{p(x)}[d(x)] - \mathbb{E}_{q(x)}[d(x)]

    """

    def __init__(self, p, q, discriminator,
                 clip_value=0.01, **kwargs):
        super().__init__(p, q, discriminator, **kwargs)
        self._clip_value = clip_value

    @property
    def loss_text(self):
        return "mean(AdversarialWD[{}||{}])".format(self._p1.prob_text,
                                                    self._p2.prob_text)

    def d_loss(self, y1, y2, *args, **kwargs):
        return - (torch.mean(y1) - torch.mean(y2))

    def g_loss(self, y1, y2, *args, **kwargs):
        if self._p1_data_dist:
            y1 = y1.detach()
        return torch.mean(y1) - torch.mean(y2)

    def train(self, train_x, **kwargs):
        loss = super().train(train_x, **kwargs)
        # Clip weights of discriminator
        for params in self.d.parameters():
            params.data.clamp_(-self._clip_value, self._clip_value)

        return loss
