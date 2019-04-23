import torch
from torch import nn
import numpy as np

from .flows import Flow
from ..utils import epsilon, sum_samples


class SqueezeLayer(Flow):
    """
    Squeeze operation.

    c * s * s -> 4c * s/2 * s/2

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([i+1 for i in range(16)]).view(1,1,4,4)
    >>> print(a)
    tensor([[[[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12],
              [13, 14, 15, 16]]]])
    >>> f = SqueezeLayer()
    >>> print(f(a))
    tensor([[[[ 1,  3],
              [ 9, 11]],
    <BLANKLINE>
             [[ 2,  4],
              [10, 12]],
    <BLANKLINE>
             [[ 5,  7],
              [13, 15]],
    <BLANKLINE>
             [[ 6,  8],
              [14, 16]]]])

    >>> print(f.inverse(f(a)))
    tensor([[[[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12],
              [13, 14, 15, 16]]]])

    """

    def __init__(self):
        super().__init__(None)
        self._logdet_jacobian = 0

    def forward(self, x, compute_jacobian=True):
        [_, channels, height, width] = x.shape

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError

        x = x.permute(0, 2, 3, 1)

        x = x.view(-1, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous().view(-1, height // 2, width // 2, channels * 4)

        z = x.permute(0, 3, 1, 2)

        return z

    def inverse(self, z):
        [_, channels, height, width] = z.shape

        if channels % 4 != 0:
            raise ValueError

        z = z.permute(0, 2, 3, 1)

        z = z.view(-1, height, width, channels // 4, 2, 2)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.contiguous().view(-1, 2 * height, 2 * width, channels // 4)

        x = z.permute(0, 3, 1, 2)

        return x


class UnsqueezeLayer(SqueezeLayer):
    """
    Unsqueeze operation.

    c * s * s -> c/4 * 2s * 2s

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([i+1 for i in range(16)]).view(1,4,2,2)
    >>> print(a)
    tensor([[[[ 1,  2],
              [ 3,  4]],
    <BLANKLINE>
             [[ 5,  6],
              [ 7,  8]],
    <BLANKLINE>
             [[ 9, 10],
              [11, 12]],
    <BLANKLINE>
             [[13, 14],
              [15, 16]]]])
    >>> f = UnsqueezeLayer()
    >>> print(f(a))
    tensor([[[[ 1,  5,  2,  6],
              [ 9, 13, 10, 14],
              [ 3,  7,  4,  8],
              [11, 15, 12, 16]]]])
    >>> print(f.inverse(f(a)))
    tensor([[[[ 1,  2],
              [ 3,  4]],
    <BLANKLINE>
             [[ 5,  6],
              [ 7,  8]],
    <BLANKLINE>
             [[ 9, 10],
              [11, 12]],
    <BLANKLINE>
             [[13, 14],
              [15, 16]]]])

    """

    def forward(self, x, compute_jacobian=True):
        return super().inverse(x)

    def inverse(self, z):
        return super().forward(z)


class PermutationLayer(Flow):
    """
    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([i+1 for i in range(16)]).view(1,4,2,2)
    >>> print(a)
    tensor([[[[ 1,  2],
              [ 3,  4]],
    <BLANKLINE>
             [[ 5,  6],
              [ 7,  8]],
    <BLANKLINE>
             [[ 9, 10],
              [11, 12]],
    <BLANKLINE>
             [[13, 14],
              [15, 16]]]])
    >>> perm = [0,3,1,2]
    >>> f = PermutationLayer(perm)
    >>> f(a)
    tensor([[[[ 1,  2],
              [ 3,  4]],
    <BLANKLINE>
             [[13, 14],
              [15, 16]],
    <BLANKLINE>
             [[ 5,  6],
              [ 7,  8]],
    <BLANKLINE>
             [[ 9, 10],
              [11, 12]]]])
    >>> f.inverse(f(a))
    tensor([[[[ 1,  2],
              [ 3,  4]],
    <BLANKLINE>
             [[ 5,  6],
              [ 7,  8]],
    <BLANKLINE>
             [[ 9, 10],
              [11, 12]],
    <BLANKLINE>
             [[13, 14],
              [15, 16]]]])

    """

    def __init__(self, permute_indices):
        super().__init__(len(permute_indices))
        self.permute_indices = permute_indices
        self.inv_permute_indices = np.argsort(self.permute_indices)
        self._logdet_jacobian = 0

    def forward(self, x, compute_jacobian=True):
        return x[:, self.permute_indices, :, :]

    def inverse(self, z):
        return z[:, self.inv_permute_indices, :, :]


class ShuffleLayer(PermutationLayer):
    def __init__(self, in_channels):
        permute_indices = np.random.permutation(in_channels)
        super().__init__(permute_indices)


class ReverseLayer(PermutationLayer):
    def __init__(self, in_channels):
        permute_indices = np.array(np.arange(0, in_channels)[::-1])
        super().__init__(permute_indices)


class BatchNorm1dFlow(Flow):
    """
    An batch normalization with the inverse transformation.

    https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py#L205

    Examples
    --------
    >>> x = torch.randn(20, 100)
    >>> f = BatchNorm1dFlow(100)
    >>> # transformation
    >>> z = f(x)
    >>> # reconstruction
    >>> _x = f.inverse(f(x))
    >>> # check this reconstruction
    >>> diff = torch.sum(torch.abs(_x-x)).data
    >>> diff < 0.1
    tensor(1, dtype=torch.uint8)
    """
    def __init__(self, in_features, momentum=0.0):
        super().__init__(in_features)
        self.log_gamma = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.momentum = momentum

        self.register_buffer('_running_mean', torch.zeros(in_features))
        self.register_buffer('_running_var', torch.ones(in_features))
        self.register_buffer('_batch_mean', torch.zeros(in_features))
        self.register_buffer('_batch_var', torch.ones(in_features))

    def forward(self, x, compute_jacobian=True):
        if self.training:
            self._batch_mean = x.mean(0)
            self._batch_var = (x - self._batch_mean).pow(2).mean(0) + epsilon()

            self._running_mean = self._running_mean * self.momentum + self._batch_mean * (1 - self.momentum)
            self._running_var = self._running_var * self.momentum + self._batch_var * (1 - self.momentum)

            mean = self._batch_mean
            var = self._batch_var
        else:
            mean = self._running_mean
            var = self._running_var

        x_hat = (x - mean) / var.sqrt()
        z = torch.exp(self.log_gamma) * x_hat + self.beta

        if compute_jacobian:
            self._logdet_jacobian = torch.sum(self.log_gamma - 0.5 * torch.log(var))

        return z

    def inverse(self, z):
        if self.training:
            mean = self._batch_mean
            var = self._batch_var
        else:
            mean = self._running_mean
            var = self._running_var

        z_hat = (z - self.beta) / torch.exp(self.log_gamma)

        x = z_hat * var.sqrt() + mean

        return x


class BatchNorm2dFlow(BatchNorm1dFlow):
    """
    An batch normalization with the inverse transformation.

    https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py#L205

    Examples
    --------
    >>> x = torch.randn(20, 100, 35, 45)
    >>> f = BatchNorm2dFlow(100)
    >>> # transformation
    >>> z = f(x)
    >>> # reconstruction
    >>> _x = f.inverse(f(x))
    >>> # check this reconstruction
    >>> diff = torch.sum(torch.abs(_x-x)).data
    >>> diff < 0.1
    tensor(1, dtype=torch.uint8)
    """
    def __init__(self, in_channels, momentum=0.0):
        super().__init__(in_channels, momentum)
        self.log_gamma = nn.Parameter(self._unsqueeze(self.log_gamma.data))
        self.beta = nn.Parameter(self._unsqueeze(self.beta.data))

        self.register_buffer('_running_mean', self._unsqueeze(self._running_mean))
        self.register_buffer('_running_var', self._unsqueeze(self._running_mean))
        self.register_buffer('_batch_mean', self._unsqueeze(self._batch_mean))
        self.register_buffer('_batch_var', self._unsqueeze(self._batch_var))

    def _unsqueeze(self, x):
        return x.unsqueeze(1).unsqueeze(2)
