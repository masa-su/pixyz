import torch
import torch.nn.functional as F
import numpy as np

from .flows import Flow
from ..utils import sum_samples


class Squeeze(Flow):
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
    >>> f = Squeeze()
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

    def forward(self, x, y=None, compute_jacobian=True):
        [_, channels, height, width] = x.shape

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError

        x = x.permute(0, 2, 3, 1)

        x = x.view(-1, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous().view(-1, height // 2, width // 2, channels * 4)

        z = x.permute(0, 3, 1, 2)

        return z

    def inverse(self, z, y=None):
        [_, channels, height, width] = z.shape

        if channels % 4 != 0:
            raise ValueError

        z = z.permute(0, 2, 3, 1)

        z = z.view(-1, height, width, channels // 4, 2, 2)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.contiguous().view(-1, 2 * height, 2 * width, channels // 4)

        x = z.permute(0, 3, 1, 2)

        return x


class Unsqueeze(Squeeze):
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
    >>> f = Unsqueeze()
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

    def forward(self, x, y=None, compute_jacobian=True):
        return super().inverse(x)

    def inverse(self, z, y=None):
        return super().forward(z)


class Permutation(Flow):
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
    >>> f = Permutation(perm)
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

    def forward(self, x, y=None, compute_jacobian=True):
        if x.dim() == 2:
            return x[:, self.permute_indices]
        elif x.dim() == 4:
            return x[:, self.permute_indices, :, :]
        raise ValueError

    def inverse(self, z, y=None):
        if z.dim() == 2:
            return z[:, self.inv_permute_indices]
        elif z.dim() == 4:
            return z[:, self.inv_permute_indices, :, :]
        raise ValueError


class Shuffle(Permutation):
    def __init__(self, in_features):
        permute_indices = np.random.permutation(in_features)
        super().__init__(permute_indices)


class Reverse(Permutation):
    def __init__(self, in_features):
        permute_indices = np.array(np.arange(0, in_features)[::-1])
        super().__init__(permute_indices)


class Flatten(Flow):
    def __init__(self, in_size=None):
        super().__init__(None)
        self.in_size = in_size
        self._logdet_jacobian = 0

    def forward(self, x, y=None, compute_jacobian=True):
        self.in_size = x.shape[1:]
        return x.view(x.size(0), -1)

    def inverse(self, z, y=None):
        if self.in_size is None:
            raise ValueError
        return z.view(z.size(0), self.in_size[0], self.in_size[1], self.in_size[2])


class Preprocess(Flow):
    def __init__(self):
        super().__init__(None)
        self.register_buffer('data_constraint', torch.tensor([0.05], dtype=torch.float32))

    @staticmethod
    def logit(x):
        return x.log() - (1. - x).log()

    def forward(self, x, y=None, compute_jacobian=True):
        # 1. transform the domain of x from [0, 1] to [0, 255]
        x = x * 255

        # 2-1. add noise to pixels to dequantize them and transform its domain ([0, 255]->[0, 1]).
        x = (x + torch.rand_like(x)) / 256.

        # 2-2. transform pixel values with logit to be unconstrained ([0, 1]->(0, 1)).
        x = (1 + (2 * x - 1) * (1 - self.data_constraint)) / 2.

        # 2-3. apply the logit function ((0, 1)->(-inf, inf)).
        z = self.logit(x)

        if compute_jacobian:
            # log-det Jacobian of transformation (2)
            logdet_jacobian = F.softplus(z) + F.softplus(-z) \
                - F.softplus(self.data_constraint.log() - (1. - self.data_constraint).log())
            logdet_jacobian = sum_samples(logdet_jacobian)

            # log-det Jacobian of transformation (1)
            logdet_jacobian = logdet_jacobian - np.log(256.) * z[0].numel()

            self._logdet_jacobian = logdet_jacobian

        return z

    def inverse(self, z, y=None):
        # transform the domain of z from (-inf, inf) to (0, 1).
        return torch.sigmoid(z)
