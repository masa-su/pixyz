import math
import torch
from torch import nn
from torch.nn import functional as F


class Flow(nn.Module):
    """Flow class. In Pixyz, all flows are required to inherit this class."""

    def __init__(self, in_features):
        """
        Parameters
        ----------
        in_features : int
            Size of each input sample

        """
        self._in_features = in_features
        self._logdet_jacobian = None

    @property
    def in_features(self):
        return self._in_features

    def forward(self, x, inverse=False, conpute_jacobian=True):
        raise NotImplementedError

    def update_jacobian(self, x):
        _ = self.forward(x, inverse=False, conpute_jacobian=True)

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`forward` or :attr:`update_jacobian` methods.

        """
        return self._logdet_jacobian


class FlowList(Flow):

    def __init__(self, flows):
        super().__init__(flows[0].in_features)
        self.flows = flows

    def forward(self, x, inverse=False, conpute_jacobian=True):
        logdet_jacobian = 0

        for flow in self.flows:
            x = flow(x, inverse, conpute_jacobian)
            if logdet_jacobian:
                logdet_jacobian += flow.logdet_jacobian

        self._logdet_jacobian = logdet_jacobian
        return x


class PlanerFlow(Flow):
    """
    Planer flow.

    .. math::
        f(x) = x + u h( w^T x + b)

    """

    def __init__(self, in_features):
        super().__init__(in_features)

        self.w = nn.Parameter(torch.Tensor(1, in_features))
        self.b = nn.Parameter(torch.Tensor(1))
        self.u = nn.Parameter(torch.Tensor(1, in_features))

        self.reset_parameters()

    def h(self, x):
        return F.tanh(x)

    def deriv_h(self, x):
        return 1 - self.h(x) ** 2

    def reset_parameters(self):
        std = 1. / math.sqrt(self.w.size(1))

        self.w.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)
        self.u.data.uniform_(-std, std)

    def forward(self, x, inverse=False, conpute_jacobian=True):
        # modify :attr:`u` so that this flow can be invertible.
        wu = torch.sum(self.w * self.b, keepdim=True)
        m_wu = -1. + F.softplus(wu)
        w_normalized = self.w / torch.norm(self.w, keepdim=True)
        u_hat = self.u + ((m_wu - wu) * w_normalized)

        # compute the flow transformation
        linear_output = F.linear(x, self.w, self.b)
        z = x + u_hat * self.h(linear_output)

        if conpute_jacobian:
            # compute the log-det Jacobian (logdet|dz/dx|)
            psi = self.deriv_h(linear_output)
            det_jacobian = 1. + torch.sum(psi * u_hat)
            logdet_jacobian = torch.log(torch.abs(det_jacobian))
            self._logdet_jacobian = logdet_jacobian

        return z

    def extra_repr(self):
        return 'in_features={}, w={}, b={}, u={}'.format(
            self.in_features, self.w, self.b, self.u is not None
        )

