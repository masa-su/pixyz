import sympy
from torch import optim, nn
import torch
from .losses import Loss
from ..utils import get_dict_values, detach_dict


class AdversarialLoss(Loss):
    def __init__(self, p, q, discriminator, input_var=None,
                 optimizer=optim.Adam, optimizer_params={}):
        if p.var != q.var:
            raise ValueError("The two distribution variables must be the same.")

        if len(p.input_var) > 0:
            self.input_dist = p
        elif len(q.input_var) > 0:
            self.input_dist = q
        else:
            raise NotImplementedError()

        super().__init__(p, q, input_var=input_var)

        self.loss_optimizer = optimizer
        self.loss_optimizer_params = optimizer_params
        self.d = discriminator

        params = discriminator.parameters()
        self.d_optimizer = optimizer(params, **optimizer_params)

    def _get_batch_n(self, x_dict):
        return get_dict_values(x_dict, self.input_dist.input_var[0])[0].shape[0]

    def d_loss(self, y_p, y_q, batch_n):
        """Evaluate a discriminator loss given outputs of the discriminator.

        Parameters
        ----------
        y_p : torch.Tensor
            Output of discriminator given sample from p.
        y_q : torch.Tensor
            Output of discriminator given sample from q.
        batch_n : int
            Batch size of inputs.

        Returns
        -------
        torch.Tensor

        """
        raise NotImplementedError()

    def g_loss(self, y_p, y_q, batch_n):
        """Evaluate a generator loss given outputs of the discriminator.

        Parameters
        ----------
        y_p : torch.Tensor
            Output of discriminator given sample from p.
        y_q : torch.Tensor
            Output of discriminator given sample from q.
        batch_n : int
            Batch size of inputs.

        Returns
        -------
        torch.Tensor

        """
        raise NotImplementedError()

    def train(self, train_x_dict, **kwargs):
        """Train the evaluation metric (discriminator).

        Parameters
        ----------
        train_x_dict : dict
            Input variables.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        loss : torch.Tensor

        """
        self.d.train()

        self.d_optimizer.zero_grad()
        loss = self.eval(train_x_dict, discriminator=True)

        # backprop
        loss.backward()

        # update params
        self.d_optimizer.step()

        return loss

    def test(self, test_x_dict, **kwargs):
        """Test the evaluation metric (discriminator).

        Parameters
        ----------
        test_x_dict : dict
            Input variables.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        loss : torch.Tensor

        """
        self.d.eval()

        with torch.no_grad():
            loss = self.eval(test_x_dict, discriminator=True)

        return loss


class AdversarialJensenShannon(AdversarialLoss):
    r"""
    Jensen-Shannon divergence (adversarial training).

    .. math::

        D_{JS}[p(x)||q(x)] \leq 2 \cdot D_{JS}[p(x)||q(x)] + 2 \log 2
         = \mathbb{E}_{p(x)}[\log d^*(x)] + \mathbb{E}_{q(x)}[\log (1-d^*(x))],

    where :math:`d^*(x) = \arg\max_{d} \mathbb{E}_{p(x)}[\log d(x)] + \mathbb{E}_{q(x)}[\log (1-d(x))]`.

    This class acts as a metric that evaluates a given distribution (generator).
    If you want to learn this evaluation metric itself, i.e., discriminator (critic), use the :class:`train` method.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Deterministic, DataDistribution, Normal
    >>> # Generator
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = nn.Linear(32, 64)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p_g = Generator()
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[32], name="p_{prior}")
    >>> p = (p_g*prior).marginalize_var("z")
    >>> print(p)
    Distribution:
      p(x) = \int p(x|z)p_{prior}(z)dz
    Network architecture:
      Normal(
        name=p_{prior}, distribution_name=Normal,
        var=['z'], cond_var=[], input_var=[], features_shape=torch.Size([32])
        (loc): torch.Size([1, 32])
        (scale): torch.Size([1, 32])
      )
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=32, out_features=64, bias=True)
      )
    >>> # Data distribution (dummy distribution)
    >>> p_data = DataDistribution(["x"])
    >>> print(p_data)
    Distribution:
      p_{data}(x)
    Network architecture:
      DataDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> # Discriminator (critic)
    >>> class Discriminator(Deterministic):
    ...     def __init__(self):
    ...         super(Discriminator, self).__init__(cond_var=["x"], var=["t"], name="d")
    ...         self.model = nn.Linear(64, 1)
    ...     def forward(self, x):
    ...         return {"t": torch.sigmoid(self.model(x))}
    >>> d = Discriminator()
    >>> print(d)
    Distribution:
      d(t|x)
    Network architecture:
      Discriminator(
        name=d, distribution_name=Deterministic,
        var=['t'], cond_var=['x'], input_var=['x'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=1, bias=True)
      )
    >>>
    >>> # Set the loss class
    >>> loss_cls = AdversarialJensenShannon(p, p_data, discriminator=d)
    >>> print(loss_cls)
    mean(D_{JS}^{Adv} \left[p(x)||p_{data}(x) \right])
    >>>
    >>> sample_x = torch.randn(2, 64) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(1.3723, grad_fn=<AddBackward0>)
    >>> # For evaluating a discriminator loss, set the `discriminator` option to True.
    >>> loss_d = loss_cls.eval({"x": sample_x}, discriminator=True)
    >>> print(loss_d) # doctest: +SKIP
    tensor(1.4990, grad_fn=<AddBackward0>)
    >>> # When training the evaluation metric (discriminator), use the train method.
    >>> train_loss = loss_cls.train({"x": sample_x})

    References
    ----------
    [Goodfellow+ 2014] Generative Adversarial Networks
    """

    def __init__(self, p, q, discriminator, input_var=None, optimizer=optim.Adam, optimizer_params={},
                 inverse_g_loss=True):
        super().__init__(p, q, discriminator,
                         input_var=input_var,
                         optimizer=optimizer, optimizer_params=optimizer_params)

        self.bce_loss = nn.BCELoss()
        self._inverse_g_loss = inverse_g_loss

    @property
    def _symbol(self):
        return sympy.Symbol("mean(D_{{JS}}^{{Adv}} \\left[{}||{} \\right])".format(self.p.prob_text,
                                                                                   self.q.prob_text))

    def _get_eval(self, x_dict, discriminator=False, **kwargs):
        batch_n = self._get_batch_n(x_dict)

        # sample x_p from p
        x_p_dict = get_dict_values(self.p.sample(x_dict, batch_n=batch_n), self.d.input_var, True)
        # sample x_q from q
        x_q_dict = get_dict_values(self.q.sample(x_dict, batch_n=batch_n), self.d.input_var, True)
        if discriminator:
            # sample y_p from d
            y_p = get_dict_values(self.d.sample(detach_dict(x_p_dict)), self.d.var)[0]
            # sample y_q from d
            y_q = get_dict_values(self.d.sample(detach_dict(x_q_dict)), self.d.var)[0]

            return self.d_loss(y_p, y_q, batch_n), x_dict

        # sample y_p from d
        y_p_dict = self.d.sample(x_p_dict)
        # sample y_q from d
        y_q_dict = self.d.sample(x_q_dict)

        y_p = get_dict_values(y_p_dict, self.d.var)[0]
        y_q = get_dict_values(y_q_dict, self.d.var)[0]

        return self.g_loss(y_p, y_q, batch_n), x_dict

    def d_loss(self, y_p, y_q, batch_n):
        # set labels
        t_p = torch.ones(batch_n, 1).to(y_p.device)
        t_q = torch.zeros(batch_n, 1).to(y_p.device)

        return self.bce_loss(y_p, t_p) + self.bce_loss(y_q, t_q)

    def g_loss(self, y_p, y_q, batch_n):
        # set labels
        t1 = torch.ones(batch_n, 1).to(y_p.device)
        t2 = torch.zeros(batch_n, 1).to(y_p.device)

        if self._inverse_g_loss:
            y_p_loss = self.bce_loss(y_p, t2)
            y_q_loss = self.bce_loss(y_q, t1)
        else:
            y_p_loss = -self.bce_loss(y_p, t1)
            y_q_loss = -self.bce_loss(y_q, t2)

        if self.p.distribution_name == "Data distribution":
            y_p_loss = y_p_loss.detach()

        if self.q.distribution_name == "Data distribution":
            y_q_loss = y_q_loss.detach()

        return y_p_loss + y_q_loss

    def train(self, train_x_dict, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict, **kwargs):
        return super().test(test_x_dict, **kwargs)


class AdversarialKullbackLeibler(AdversarialLoss):
    r"""
    Kullback-Leibler divergence (adversarial training).

    .. math::

        D_{KL}[p(x)||q(x)] = \mathbb{E}_{p(x)}[\log \frac{p(x)}{q(x)}]
         \approx \mathbb{E}_{p(x)}[\log \frac{d^*(x)}{1-d^*(x)}],

    where :math:`d^*(x) = \arg\max_{d} \mathbb{E}_{q(x)}[\log d(x)] + \mathbb{E}_{p(x)}[\log (1-d(x))]`.

    Note that this divergence is minimized to close :math:`p` to :math:`q`.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Deterministic, DataDistribution, Normal
    >>> # Generator
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = nn.Linear(32, 64)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p_g = Generator()
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[32], name="p_{prior}")
    >>> p = (p_g*prior).marginalize_var("z")
    >>> print(p)
    Distribution:
      p(x) = \int p(x|z)p_{prior}(z)dz
    Network architecture:
      Normal(
        name=p_{prior}, distribution_name=Normal,
        var=['z'], cond_var=[], input_var=[], features_shape=torch.Size([32])
        (loc): torch.Size([1, 32])
        (scale): torch.Size([1, 32])
      )
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=32, out_features=64, bias=True)
      )
    >>> # Data distribution (dummy distribution)
    >>> p_data = DataDistribution(["x"])
    >>> print(p_data)
    Distribution:
      p_{data}(x)
    Network architecture:
      DataDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> # Discriminator (critic)
    >>> class Discriminator(Deterministic):
    ...     def __init__(self):
    ...         super(Discriminator, self).__init__(cond_var=["x"], var=["t"], name="d")
    ...         self.model = nn.Linear(64, 1)
    ...     def forward(self, x):
    ...         return {"t": torch.sigmoid(self.model(x))}
    >>> d = Discriminator()
    >>> print(d)
    Distribution:
      d(t|x)
    Network architecture:
      Discriminator(
        name=d, distribution_name=Deterministic,
        var=['t'], cond_var=['x'], input_var=['x'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=1, bias=True)
      )
    >>>
    >>> # Set the loss class
    >>> loss_cls = AdversarialKullbackLeibler(p, p_data, discriminator=d)
    >>> print(loss_cls)
    mean(D_{KL}^{Adv} \left[p(x)||p_{data}(x) \right])
    >>>
    >>> sample_x = torch.randn(2, 64) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> # The evaluation value might be negative if the discriminator training is incomplete.
    >>> print(loss) # doctest: +SKIP
    tensor(-0.8377, grad_fn=<AddBackward0>)
    >>> # For evaluating a discriminator loss, set the `discriminator` option to True.
    >>> loss_d = loss_cls.eval({"x": sample_x}, discriminator=True)
    >>> print(loss_d) # doctest: +SKIP
    tensor(1.9321, grad_fn=<AddBackward0>)
    >>> # When training the evaluation metric (discriminator), use the train method.
    >>> train_loss = loss_cls.train({"x": sample_x})

    References
    ----------
    [Kim+ 2018] Disentangling by Factorising
    """

    def __init__(self, p, q, discriminator, **kwargs):
        super().__init__(p, q, discriminator, **kwargs)
        self.bce_loss = nn.BCELoss()

    @property
    def _symbol(self):
        return sympy.Symbol("mean(D_{{KL}}^{{Adv}} \\left[{}||{} \\right])".format(self.p.prob_text,
                                                                                   self.q.prob_text))

    def _get_eval(self, x_dict, discriminator=False, **kwargs):
        batch_n = self._get_batch_n(x_dict)

        # sample x_p from p
        x_p_dict = get_dict_values(self.p.sample(x_dict, batch_n=batch_n), self.d.input_var, True)

        if discriminator:
            # sample x_q from q
            x_q_dict = get_dict_values(self.q.sample(x_dict, batch_n=batch_n), self.d.input_var, True)

            # sample y_p from d
            y_p = get_dict_values(self.d.sample(detach_dict(x_p_dict)), self.d.var)[0]
            # sample y_q from d
            y_q = get_dict_values(self.d.sample(detach_dict(x_q_dict)), self.d.var)[0]

            return self.d_loss(y_p, y_q, batch_n), x_dict

        # sample y from d
        y_p = get_dict_values(self.d.sample(x_p_dict), self.d.var)[0]

        return self.g_loss(y_p, batch_n), x_dict

    def g_loss(self, y_p, batch_n):
        """Evaluate a generator loss given an output of the discriminator.

        Parameters
        ----------
        y_p : torch.Tensor
            Output of discriminator given sample from p.
        batch_n : int
            Batch size of inputs.

        Returns
        -------
        torch.Tensor

        """
        # set labels
        t_p = torch.ones(batch_n, 1).to(y_p.device)
        t_q = torch.zeros(batch_n, 1).to(y_p.device)

        y_p_loss = -self.bce_loss(y_p, t_p) + self.bce_loss(y_p, t_q)  # log (y_p) - log (1 - y_p)

        return y_p_loss

    def d_loss(self, y_p, y_q, batch_n):
        # set labels
        t_p = torch.ones(batch_n, 1).to(y_p.device)
        t_q = torch.zeros(batch_n, 1).to(y_p.device)

        return self.bce_loss(y_p, t_p) + self.bce_loss(y_q, t_q)

    def train(self, train_x_dict, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict, **kwargs):
        return super().test(test_x_dict, **kwargs)


class AdversarialWassersteinDistance(AdversarialJensenShannon):
    r"""
    Wasserstein distance (adversarial training).

    .. math::

         W(p, q) = \sup_{||d||_{L} \leq 1} \mathbb{E}_{p(x)}[d(x)] - \mathbb{E}_{q(x)}[d(x)]

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Deterministic, DataDistribution, Normal
    >>> # Generator
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = nn.Linear(32, 64)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p_g = Generator()
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[32], name="p_{prior}")
    >>> p = (p_g*prior).marginalize_var("z")
    >>> print(p)
    Distribution:
      p(x) = \int p(x|z)p_{prior}(z)dz
    Network architecture:
      Normal(
        name=p_{prior}, distribution_name=Normal,
        var=['z'], cond_var=[], input_var=[], features_shape=torch.Size([32])
        (loc): torch.Size([1, 32])
        (scale): torch.Size([1, 32])
      )
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=32, out_features=64, bias=True)
      )
    >>> # Data distribution (dummy distribution)
    >>> p_data = DataDistribution(["x"])
    >>> print(p_data)
    Distribution:
      p_{data}(x)
    Network architecture:
      DataDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> # Discriminator (critic)
    >>> class Discriminator(Deterministic):
    ...     def __init__(self):
    ...         super(Discriminator, self).__init__(cond_var=["x"], var=["t"], name="d")
    ...         self.model = nn.Linear(64, 1)
    ...     def forward(self, x):
    ...         return {"t": self.model(x)}
    >>> d = Discriminator()
    >>> print(d)
    Distribution:
      d(t|x)
    Network architecture:
      Discriminator(
        name=d, distribution_name=Deterministic,
        var=['t'], cond_var=['x'], input_var=['x'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=1, bias=True)
      )
    >>>
    >>> # Set the loss class
    >>> loss_cls = AdversarialWassersteinDistance(p, p_data, discriminator=d)
    >>> print(loss_cls)
    mean(W^{Adv} \left(p(x), p_{data}(x) \right))
    >>>
    >>> sample_x = torch.randn(2, 64) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(-0.0060, grad_fn=<SubBackward0>)
    >>> # For evaluating a discriminator loss, set the `discriminator` option to True.
    >>> loss_d = loss_cls.eval({"x": sample_x}, discriminator=True)
    >>> print(loss_d) # doctest: +SKIP
    tensor(-0.3802, grad_fn=<NegBackward>)
    >>> # When training the evaluation metric (discriminator), use the train method.
    >>> train_loss = loss_cls.train({"x": sample_x})

    References
    ----------
    [Arjovsky+ 2017] Wasserstein GAN
    """

    def __init__(self, p, q, discriminator,
                 clip_value=0.01, **kwargs):
        super().__init__(p, q, discriminator, **kwargs)
        self._clip_value = clip_value

    @property
    def _symbol(self):
        return sympy.Symbol("mean(W^{{Adv}} \\left({}, {} \\right))".format(self.p.prob_text, self.q.prob_text))

    def d_loss(self, y_p, y_q, *args, **kwargs):
        return - (torch.mean(y_p) - torch.mean(y_q))

    def g_loss(self, y_p, y_q, *args, **kwargs):
        if self.p.distribution_name == "Data distribution":
            y_p = y_p.detach()

        if self.q.distribution_name == "Data distribution":
            y_q = y_q.detach()
        return torch.mean(y_p) - torch.mean(y_q)

    def train(self, train_x_dict, **kwargs):
        loss = super().train(train_x_dict, **kwargs)

        # Clip weights of discriminator
        for params in self.d.parameters():
            params.data.clamp_(-self._clip_value, self._clip_value)

        return loss

    def test(self, test_x_dict, **kwargs):
        return super().test(test_x_dict, **kwargs)
