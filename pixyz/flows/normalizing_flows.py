import math
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import epsilon
from .flows import Flow


class PlanarFlow(Flow):
    r"""
    Planar flow.

    .. math::
        f(\mathbf{x}) = \mathbf{x} + \mathbf{u} h( \mathbf{w}^T \mathbf{x} + \mathbf{b})

    """

    def __init__(self, in_features, constraint_u=False):
        super().__init__(in_features)

        self.w = nn.Parameter(torch.Tensor(1, in_features))
        self.b = nn.Parameter(torch.Tensor(1))
        self.u = nn.Parameter(torch.Tensor(1, in_features))

        self.reset_parameters()
        self.constraint_u = constraint_u

    def deriv_tanh(self, x):
        return 1 - torch.tanh(x) ** 2

    def reset_parameters(self):
        std = 1. / math.sqrt(self.w.size(1))

        self.w.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)
        self.u.data.uniform_(-std, std)

    def forward(self, x, y=None, compute_jacobian=True):
        if self.constraint_u:
            # modify :attr:`u` so that this flow can be invertible.
            wu = torch.mm(self.w, self.u.t())  # (1, 1)
            m_wu = -1. + F.softplus(wu)
            w_normalized = self.w / torch.norm(self.w, keepdim=True)
            u_hat = self.u + ((m_wu - wu) * w_normalized)  # (1, in_features)
        else:
            u_hat = self.u

        # compute the flow transformation
        linear_output = F.linear(x, self.w, self.b)  # (n_batch, 1)
        z = x + u_hat * torch.tanh(linear_output)

        if compute_jacobian:
            # compute the log-det Jacobian (logdet|dz/dx|)
            psi = self.deriv_tanh(linear_output) * self.w  # (n_batch, in_features)
            det_jacobian = 1. + torch.mm(psi, u_hat.t()).squeeze()  # (n_batch, 1) -> (n_batch)
            logdet_jacobian = torch.log(torch.abs(det_jacobian) + epsilon())
            self._logdet_jacobian = logdet_jacobian

        return z

    def inverse(self, z, y=None):
        raise NotImplementedError()

    def extra_repr(self):
        return 'in_features={}, constraint_u={}'.format(
            self.in_features, self.constraint_u
        )
