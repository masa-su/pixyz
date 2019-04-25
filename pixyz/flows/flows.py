import math
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import epsilon


class Flow(nn.Module):
    """Flow class. In Pixyz, all flows are required to inherit this class."""

    def __init__(self, in_features):
        """
        Parameters
        ----------
        in_features : int
            Size of each input sample

        """
        super().__init__()
        self._in_features = in_features
        self._logdet_jacobian = None

    @property
    def in_features(self):
        return self._in_features

    def forward(self, x, compute_jacobian=True):
        """
        Parameters
        ----------
        x : torch.Tensor
        compute_jacobian : bool

        Returns
        -------
        z : torch.Tensor

        """
        z = x
        return z

    def inverse(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor

        Returns
        -------
        x : torch.Tensor

        """
        x = z
        return x

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`forward` or :attr:`update_jacobian` methods.

        """
        return self._logdet_jacobian


class FlowList(Flow):

    def __init__(self, flow_list):
        """
        Parameters
        ----------
        flow_list : list
        """

        super().__init__(flow_list[0].in_features)
        self.flow_list = nn.ModuleList(flow_list)

    def forward(self, x, compute_jacobian=True):
        logdet_jacobian = 0

        for flow in self.flow_list:
            x = flow.forward(x, compute_jacobian)
            if compute_jacobian:
                logdet_jacobian = logdet_jacobian + flow.logdet_jacobian

        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian

        return x

    def inverse(self, z):
        for flow in self.flow_list[::-1]:
            z = flow.inverse(z)
        return z

    def __repr__(self):
        # rename "ModuleList" to "FlowList"
        flow_list_repr = self.flow_list.__repr__()[10:]
        flow_list_repr = "FlowList" + flow_list_repr
        return flow_list_repr


class PlanarFlow(Flow):
    """
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

    def forward(self, x, compute_jacobian=True):
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

    def extra_repr(self):
        return 'in_features={}, constraint_u={}'.format(
            self.in_features, self.constraint_u
        )
