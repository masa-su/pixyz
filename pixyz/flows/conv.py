import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import scipy as sp

from .flows import Flow


class ChannelConv(Flow):
    """
    Invertible 1 × 1 convolution.

    Notes
    -----
    This is implemented with reference to the following code.
    https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    """

    def __init__(self, in_channels, decomposed=False):
        super().__init__(in_channels)
        w_shape = [in_channels, in_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            # LU decomposition
            np_p, np_l, np_u = sp.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))

            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.decomposed = decomposed

    def get_parameters(self, x, inverse):
        w_shape = self.w_shape
        pixels = np.prod(x.size()[2:])
        device = x.device

        if not self.decomposed:
            logdet_jacobian = torch.slogdet(self.weight.cpu())[1].to(device) * pixels
            if not inverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)
            return weight, logdet_jacobian
        else:
            self.p = self.p.to(device)
            self.sign_s = self.sign_s.to(device)
            self.l_mask = self.l_mask.to(device)
            self.eye = self.eye.to(device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            logdet_jacobian = torch.sum(self.log_s) * pixels
            if not inverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), logdet_jacobian

    def forward(self, x, y=None, compute_jacobian=True):
        weight, logdet_jacobian = self.get_parameters(x, inverse=False)
        z = F.conv2d(x, weight)

        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian

        return z

    def inverse(self, x, y=None):
        weight, _ = self.get_parameters(x, inverse=True)
        z = F.conv2d(x, weight)
        return z
