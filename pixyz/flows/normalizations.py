import torch
from torch import nn
import numpy as np

from .flows import Flow
from ..utils import epsilon


class BatchNorm1d(Flow):
    """
    A batch normalization with the inverse transformation.

    Notes
    -----
    This is implemented with reference to the following code.
    https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py#L205

    Examples
    --------
    >>> x = torch.randn(20, 100)
    >>> f = BatchNorm1d(100)
    >>> # transformation
    >>> z = f(x)
    >>> # reconstruction
    >>> _x = f.inverse(f(x))
    >>> # check this reconstruction
    >>> diff = torch.sum(torch.abs(_x-x)).item()
    >>> diff < 0.1
    True
    """
    def __init__(self, in_features, momentum=0.0):
        super().__init__(in_features)

        self.log_gamma = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(in_features))
        self.register_buffer('running_var', torch.ones(in_features))

    def forward(self, x, y=None, compute_jacobian=True):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x - self.batch_mean).pow(2).mean(0) + epsilon()

            self.running_mean = self.running_mean * self.momentum
            self.running_var = self.running_var * self.momentum

            self.running_mean = self.running_mean + (self.batch_mean.data * (1 - self.momentum))
            self.running_var = self.running_var + (self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var

        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / var.sqrt()
        z = torch.exp(self.log_gamma) * x_hat + self.beta

        if compute_jacobian:
            self._logdet_jacobian = (self.log_gamma - 0.5 * torch.log(var)).sum(-1)

        return z

    def inverse(self, z, y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (z - self.beta) / torch.exp(self.log_gamma)

        x = x_hat * var.sqrt() + mean

        return x


class BatchNorm2d(BatchNorm1d):
    """
    A batch normalization with the inverse transformation.

    Notes
    -----
    This is implemented with reference to the following code.
    https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py#L205

    Examples
    --------
    >>> x = torch.randn(20, 100, 35, 45)
    >>> f = BatchNorm2d(100)
    >>> # transformation
    >>> z = f(x)
    >>> # reconstruction
    >>> _x = f.inverse(f(x))
    >>> # check this reconstruction
    >>> diff = torch.sum(torch.abs(_x-x)).item()
    >>> diff < 0.1
    True
    """
    def __init__(self, in_features, momentum=0.0):
        super().__init__(in_features, momentum)
        self.log_gamma = nn.Parameter(self._unsqueeze(self.log_gamma.data))
        self.beta = nn.Parameter(self._unsqueeze(self.beta.data))

        self.register_buffer('running_mean', self._unsqueeze(self.running_mean))
        self.register_buffer('running_var', self._unsqueeze(self.running_var))

    def _unsqueeze(self, x):
        return x.unsqueeze(1).unsqueeze(2)


class ActNorm2d(Flow):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.

    Notes
    -----
    This is implemented with reference to the following code.
    https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    """

    def __init__(self, in_features, scale=1.):
        super().__init__(in_features)
        # register mean and scale
        size = [1, in_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.scale = float(scale)
        self.inited = False

    def initialize_parameters(self, x):
        if not self.training:
            return
        assert x.device == self.bias.device
        with torch.no_grad():
            bias = torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = torch.mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + epsilon()))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, x, inverse=False):
        if not inverse:
            return x + self.bias
        else:
            return x - self.bias

    def _scale(self, x, compute_jacobian=True, inverse=False):
        logs = self.logs
        if not inverse:
            x = x * torch.exp(logs)
        else:
            x = x * torch.exp(-logs)
        if compute_jacobian:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            pixels = np.prod(x.size()[2:])
            logdet_jacobian = torch.sum(logs) * pixels

            return x, logdet_jacobian

        return x, None

    def forward(self, x, y=None, compute_jacobian=True):
        if not self.inited:
            self.initialize_parameters(x)

        # center and scale
        x = self._center(x, inverse=False)
        x, logdet_jacobian = self._scale(x, compute_jacobian, inverse=False)
        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian

        return x

    def inverse(self, x, y=None):
        if not self.inited:
            self.initialize_parameters(x)

        # scale and center
        x, _ = self._scale(x, compute_jacobian=False, inverse=True)
        x = self._center(x, inverse=True)
        return x
