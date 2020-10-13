from torch import optim

from ..models.model import Model
from ..losses import AdversarialJensenShannon
from ..distributions import DataDistribution


class GAN(Model):
    r"""
    Generative Adversarial Network

    (Adversarial) Jensen-Shannon divergence between given distributions (p_data, p)
    is set as the loss class of this model.

    Examples
    --------
    >>> import torch
    >>> from torch import nn, optim
    >>> from pixyz.distributions import Deterministic
    >>> from pixyz.distributions import Normal
    >>> from pixyz.models import GAN
    >>> from pixyz.utils import print_latex
    >>> x_dim = 128
    >>> z_dim = 100
    ...
    >>> # Set distributions (Distribution API)
    ...
    >>> # generator model p(x|z)
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = nn.Sequential(
    ...             nn.Linear(z_dim, x_dim),
    ...             nn.Sigmoid()
    ...         )
    ...     def forward(self, z):
    ...         x = self.model(z)
    ...         return {"x": x}
    ...
    >>> # prior model p(z)
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[z_dim], name="p_{prior}")
    ...
    >>> # generative model
    >>> p_g = Generator()
    >>> p = (p_g*prior).marginalize_var("z")
    ...
    >>> # discriminator model p(t|x)
    >>> class Discriminator(Deterministic):
    ...     def __init__(self):
    ...         super(Discriminator, self).__init__(cond_var=["x"], var=["t"], name="d")
    ...         self.model = nn.Sequential(
    ...             nn.Linear(x_dim, 1),
    ...             nn.Sigmoid()
    ...         )
    ...     def forward(self, x):
    ...         t = self.model(x)
    ...         return {"t": t}
    ...
    >>> d = Discriminator()
    >>> # Set a model (Model API)
    >>> model = GAN(p, d, optimizer_params={"lr":0.0002}, d_optimizer_params={"lr":0.0002})
    >>> print(model)
    Distributions (for training):
      p(x)
    Loss function:
      mean(D_{JS}^{Adv} \left[p_{data}(x)||p(x) \right])
    Optimizer:
      Adam (
      Parameter Group 0
          amsgrad: False
          betas: (0.9, 0.999)
          eps: 1e-08
          lr: 0.0002
          weight_decay: 0
      )
    >>> # Train and test the model
    >>> data = torch.randn(1, x_dim)  # Pseudo data
    >>> train_loss = model.train({"x": data})
    >>> test_loss = model.test({"x": data})
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
            d_loss = self.loss_cls.loss_train(train_x_dict, **kwargs)
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
            d_loss = self.loss_cls.loss_test(test_x_dict, **kwargs)
            return loss, d_loss
        return loss
