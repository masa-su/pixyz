from __future__ import print_function
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_dict_values, epsilon
from .distributions import Distribution


class Flow(Distribution):
    def __init__(self, prior, dim, num_layers=1, var=[],
                 flow_layer=None, flow_name=None, name="p"):
        super().__init__(cond_var=prior.cond_var, var=var,
                         name=name, dim=dim)
        self.prior = prior
        self.flows = nn.ModuleList([flow_layer(dim)
                                    for _ in range(num_layers)])
        self._flow_name = flow_name



    @property
    def prob_text(self):
        _var_text = []
        _text = "{}={}({})".format(','.join(self._var),
                                   self._flow_name,
                                   ','.join(self.prior.var))
        _var_text += [_text]
        if len(self._cond_var) != 0:
            _var_text += [','.join(self._cond_var)]

        _prob_text = "{}({})".format(
            self._name,
            "|".join(_var_text)
        )

        return _prob_text

    def forward(self, x, jacobian=False):
        if jacobian is False:
            for flow in self.flows:
                x = flow(x)
            output = x

        else:
            logdet_jacobian = 0
            for flow in self.flows:
                x, _logdet_jacobian = flow(x, jacobian)
                logdet_jacobian += _logdet_jacobian
            output = logdet_jacobian

        return output

    def sample(self, x={}, only_flow=False, return_all=True, **kwargs):
        if only_flow:
            _samples = get_dict_values(x, self.var)
        else:
            samples_dict = self.prior.sample(x, **kwargs)
            _samples = get_dict_values(samples_dict, self.prior.var)
        output = self.forward(_samples[0], jacobian=False)
        output_dict = {self.var[0]: output}

        if return_all:
            output_dict.update(samples_dict)

        return output_dict

    def log_likelihood(self, x):
        log_dist = self.prior.log_likelihood(x)

        x_values = get_dict_values(x, self.prior.var)
        logdet_jacobian = self.forward(x_values[0], jacobian=True)

        return log_dist - logdet_jacobian


class PlanarFlow(Flow):
    def __init__(self, prior, dim, num_layers=1,
                 var=[], **kwargs):
        super(PlanarFlow, self).__init__(prior, dim,
                                         num_layers=num_layers,
                                         var=var,
                                         flow_layer=PlanarFlowLayer,
                                         flow_name="PlanarFlow", **kwargs)


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
