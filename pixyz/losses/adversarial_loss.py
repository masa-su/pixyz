from torch import optim, nn
import torch
from .losses import Loss
from ..utils import get_dict_values, detach_dict


class AdversarialLoss(Loss):
    def __init__(self, p, q, discriminator, input_var=None,
                 optimizer=optim.Adam, optimizer_params={}):
        super().__init__(p, q, input_var=input_var)
        self.loss_optimizer = optimizer
        self.loss_optimizer_params = optimizer_params
        self.d = discriminator

        params = discriminator.parameters()
        self.d_optimizer = optimizer(params, **optimizer_params)

        # TODO: fix this decision rule
        if p.distribution_name == "Data distribution":
            self._p_data_dist = True
        else:
            self._p_data_dist = False

        if q.distribution_name == "Data distribution":
            self._q_data_dist = True
        else:
            self._q_data_dist = False

    def d_loss(self, y_p, y_q, batch_size):
        raise NotImplementedError

    def g_loss(self, y_p, y_q, batch_size):
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
        return "mean(AdversarialJS[{}||{}])".format(self._p.prob_text,
                                                    self._q.prob_text)

    def _get_estimated_value(self, x, discriminator=False, **kwargs):
        batch_size = get_dict_values(x, self._p.input_var[0])[0].shape[0]

        # sample x from p
        x_dict = get_dict_values(x, self._p.input_var, True)
        if self._p_data_dist:
            x_p_dict = x_dict
        else:
            x_p_dict = self._p.sample(x_dict, batch_size=batch_size)
            x_p_dict = get_dict_values(x_p_dict, self.d.input_var, True)

        # sample x from q
        x_dict = get_dict_values(x, self._q.input_var, True)
        x_q_dict = self._q.sample(x_dict, batch_size=batch_size)
        x_q_dict = get_dict_values(x_q_dict, self.d.input_var, True)

        if discriminator:
            # sample y from x_p
            y_p_dict = self.d.sample(detach_dict(x_p_dict))
            y_p = get_dict_values(y_p_dict, self.d.var)[0]

            # sample y from x_q
            y_q_dict = self.d.sample(detach_dict(x_q_dict))
            y_q = get_dict_values(y_q_dict, self.d.var)[0]

            return self.d_loss(y_p, y_q, batch_size), x

        # sample y from x_p
        y_p_dict = self.d.sample(x_p_dict)
        # sample y from x_q
        y_q_dict = self.d.sample(x_q_dict)

        y_p = get_dict_values(y_p_dict, self.d.var)[0]
        y_q = get_dict_values(y_q_dict, self.d.var)[0]

        return self.g_loss(y_p, y_q, batch_size), x  # TODO: fix

    def d_loss(self, y_p, y_q, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y_p.device)
        t2 = torch.zeros(batch_size, 1).to(y_p.device)
        return self.bce_loss(y_p, t1) + self.bce_loss(y_q, t2)

    def g_loss(self, y_p, y_q, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y_p.device)
        t2 = torch.zeros(batch_size, 1).to(y_p.device)

        if self._inverse_g_loss:
            y_p_loss = self.bce_loss(y_p, t2)
            y_q_loss = self.bce_loss(y_q, t1)
        else:
            y_p_loss = -self.bce_loss(y_p, t1)
            y_q_loss = -self.bce_loss(y_q, t2)

        if self._p_data_dist:
            y_p_loss = y_p_loss.detach()

        return y_p_loss + y_q_loss


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
        return "mean(AdversarialKL[{}||{}])".format(self._p.prob_text,
                                                    self._q.prob_text)

    def _get_estimated_value(self, x, discriminator=False, **kwargs):
        batch_size = get_dict_values(x, self._p.input_var[0])[0].shape[0]

        # sample x from p
        x_dict = get_dict_values(x, self._p.input_var, True)
        x_p_dict = self._p.sample(x_dict, batch_size=batch_size)
        x_p_dict = get_dict_values(x_p_dict, self.d.input_var, True)

        if discriminator:
            # sample x from q
            x_dict = get_dict_values(x, self._q.input_var, True)
            x_q_dict = self._q.sample(x_dict, batch_size=batch_size)
            x_q_dict = get_dict_values(x_q_dict, self.d.input_var, True)

            # sample y_p from d
            y_p_dict = self.d.sample(detach_dict(x_p_dict))
            y_p = get_dict_values(y_p_dict, self.d.var)[0]

            # sample y_q from d
            y_q_dict = self.d.sample(detach_dict(x_q_dict))
            y_q = get_dict_values(y_q_dict, self.d.var)[0]

            return self.d_loss(y_p, y_q, batch_size), x

        # sample y from d
        y_p_dict = self.d.sample(x_p_dict)
        y_p = get_dict_values(y_p_dict, self.d.var)[0]

        return self.g_loss(y_p, batch_size), x

    def g_loss(self, y_p, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y_p.device)
        t2 = torch.zeros(batch_size, 1).to(y_p.device)

        y_p_loss = -self.bce_loss(y_p, t1) + self.bce_loss(y_p, t2)

        return y_p_loss

    def d_loss(self, y_p, y_q, batch_size):
        # set labels
        t1 = torch.ones(batch_size, 1).to(y_p.device)
        t2 = torch.zeros(batch_size, 1).to(y_p.device)
        return self.bce_loss(y_p, t1) + self.bce_loss(y_q, t2)


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
        return "mean(AdversarialWD[{}||{}])".format(self._p.prob_text,
                                                    self._q.prob_text)

    def d_loss(self, y_p, y_q, *args, **kwargs):
        return - (torch.mean(y_p) - torch.mean(y_q))

    def g_loss(self, y_p, y_q, *args, **kwargs):
        if self._p_data_dist:
            y_p = y_p.detach()
        return torch.mean(y_p) - torch.mean(y_q)

    def train(self, train_x, **kwargs):
        loss = super().train(train_x, **kwargs)
        # Clip weights of discriminator
        for params in self.d.parameters():
            params.data.clamp_(-self._clip_value, self._clip_value)

        return loss
