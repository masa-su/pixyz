from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_dict_values


class RealNVP(nn.Module):
    def __init__(self, dist, in_features,
                 hidden_features=512,
                 num_layers=2, var=[]):
        super(RealNVP, self).__init__()
        self.dist = dist
        self.var = var  # x
        self.cond_var = self.dist.cond_var
        self.var_dist = self.dist.var  # z

        flow_list =\
            [AffineCouplingLayer(in_features,
                                 hidden_features=hidden_features,
                                 num_layers=num_layers,
                                 pattern=0),
             AffineCouplingLayer(in_features,
                                 hidden_features=hidden_features,
                                 num_layers=num_layers,
                                 pattern=1),
             AffineCouplingLayer(in_features,
                                 hidden_features=hidden_features,
                                 num_layers=num_layers,
                                 pattern=0),
             AffineCouplingLayer(in_features,
                                 hidden_features=hidden_features,
                                 num_layers=num_layers,
                                 pattern=1)]

        self.flows = nn.ModuleList(flow_list)
        self.flow_name = "RealNVP"

        self.prob_text = "{}({} ; {})".format(
            self.flow_name,
            ','.join(var),
            dist.prob_text
        )
        self.prob_factorized_text = self.prob_text

    def forward(self, x, inverse=False, jacobian=False):
        if inverse is False:
            _flows = self.flows
        else:
            _flows = self.flows[::-1]

        if jacobian is False:
            for flow in _flows:
                x = flow(x, inverse=inverse)
            output = x

        else:
            logdet_jacobian = 0
            for flow in _flows:
                x, _logdet_jacobian = flow(x,
                                           inverse=inverse,
                                           jacobian=jacobian)
                logdet_jacobian += _logdet_jacobian
            output = logdet_jacobian

        return output

    def sample(self, x=None, only_flow=False, **kwargs):
        # x~p()
        if only_flow:
            samples = x
        else:
            samples = self.dist.sample(x, **kwargs)
        _samples = get_dict_values(samples, self.var_dist)
        output = self.forward(_samples[0],
                              inverse=True, jacobian=False)

        samples[self.var[0]] = output
        return samples

    def sample_inv(self, x, **kwargs):
        # z~p(x)
        samples = x
        _samples = get_dict_values(x, self.var)
        output = self.forward(_samples[0], jacobian=False)

        samples[self.var[0]] = output
        return samples

    def log_likelihood(self, x):
        # use a bijection function
        # z=f(x)
        _x = get_dict_values(x, self.var)
        samples = self.forward(_x[0])
        # log p(z)
        log_dist = self.dist.log_likelihood({self.var_dist[0]: samples})

        # df(x)/dx
        x_values = get_dict_values(x, self.var)
        logdet_jacobian = self.forward(x_values[0], jacobian=True)

        return log_dist + logdet_jacobian


class AffineCouplingLayer(nn.Module):
    # TODO: Conv ver. (checkerboard and channel)
    def __init__(self, in_features,
                 hidden_features=512,
                 num_layers=2,
                 masked_type="checkerboard",
                 pattern=0  # 0 or 1
                 ):
        super(AffineCouplingLayer, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features

        flow_list = [nn.Linear(in_features,
                               hidden_features)]
        flow_list += [nn.Linear(hidden_features,
                                hidden_features)
                      for _ in range(num_layers-1)]
        flow_list += [nn.Linear(hidden_features,
                                2 * in_features)]

        self.flows = nn.ModuleList(flow_list)

        self.masked_type = masked_type
        self.pattern = pattern

    def _scale_translation(self, x):
        for flow in self.flows[:-1]:
            x = F.relu(flow(x))
        x = self.flows[-1](x)
        scale, trans = torch.chunk(x, chunks=2, dim=-1)

        scale = self.masking(scale, True)
        trans = self.masking(trans, True)
        return scale, trans

    def masking(self, x, reverse=False):
        x_shape = x.shape
        if self.masked_type == "checkerboard":
            mask = torch.zeros(x_shape[1])
            mask[self.pattern::2] = 1
        else:
            NotImplementedError

        if reverse:
            return x * (1 - mask).unsqueeze(0)
        else:
            return x * mask.unsqueeze(0)

    def forward(self, x, inverse=False, jacobian=False):
        # forward: self.var -> self.var_dist (x->z)
        # inverse: self.var_dist -> self.var (z->x)

        x_0 = self.masking(x, False)
        x_1 = self.masking(x, True)
        scale, trans = self._scale_translation(x_0)

        if inverse:
            x_1 = (x_1 - trans) * torch.exp(-scale)
        else:
            x_1 = x_1 * torch.exp(scale) + trans

        output = x_0 + x_1

        if jacobian:
            logdet_jacobian = torch.sum(scale, dim=-1)
            return output, logdet_jacobian

        return output

    def extra_repr(self):
        return 'in_features={}, '\
               'hidden_features={}'.format(self.in_features,
                                           self.hidden_features)
