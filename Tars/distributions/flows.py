from __future__ import print_function
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_dict_values, epsilon


class Flow(nn.Module):
    def __init__(self, dist, in_features, num_layers=1, var=[],
                 flow_layer=None, flow_name=None):
        super(Flow, self).__init__()
        self.dist = dist
        self.var = var
        self.cond_var = self.dist.cond_var
        self.var_dist = self.dist.var
        self.flows = nn.ModuleList([flow_layer(in_features)
                                    for _ in range(num_layers)])
        self.flow_name = flow_name

        self.prob_text = "{}({} ; {})".format(
            flow_name,
            ','.join(var),
            dist.prob_text
        )
        self.prob_factorized_text = self.prob_text

    def forward(self, x, jacobian=False):
        if jacobian is False:
            for i, flow in enumerate(self.flows):
                x = flow(x)
            output = x

        else:
            logdet_jacobian = 0
            for i, flow in enumerate(self.flows):
                x, _logdet_jacobian = flow(x, jacobian)
                logdet_jacobian += _logdet_jacobian
            output = logdet_jacobian

        return output

    def sample(self, x):
        samples = self.dist.sample(x)
        _samples = get_dict_values(samples, self.var_dist)
        output = self.forward(_samples[0], jacobian=False)

        samples[self.var[0]] = output
        return samples

    def log_likelihood(self, x):
        log_dist = self.dist.log_likelihood(x)

        x_values = get_dict_values(x, self.var_dist)
        logdet_jacobian = self.forward(x_values[0], jacobian=True)

        return log_dist - logdet_jacobian


class PlanarFlow(Flow):
    def __init__(self, dist, in_features, num_layers=1,
                 var=[]):
        super(PlanarFlow, self).__init__(dist, in_features,
                                         num_layers=num_layers,
                                         var=var,
                                         flow_layer=PlanarFlowLayer,
                                         flow_name="PlanarFlow")


class PlanarFlowLayer(nn.Module):
    def __init__(self, in_features):
        super(PlanarFlowLayer, self).__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.u = nn.Parameter(torch.Tensor(1, in_features))

        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.u.data.uniform_(-stdv, stdv)

    def forward(self, x, jacobian=False):
        z = F.tanh(F.linear(x, self.weight, self.bias))
        output = x + self.u * z

        if jacobian:
            # TODO: use autograd
            z_grad = (1 - z ** 2)
            psi = z_grad * self.weight
            det_grad = 1 + torch.mm(psi, self.u.t())
            logdet_jacobian = torch.log(torch.abs(det_grad) + epsilon())

            return output, logdet_jacobian

        return output

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)
