from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_dict_values


class RealNVP(nn.Module):
    def __init__(self, dist, in_features, masked_features,
                 hidden_features=512,
                 num_layers=2, var=[]):
        super(RealNVP, self).__init__()
        self.dist = dist
        self.var = var  # x
        self.cond_var = self.dist.cond_var
        self.var_dist = self.dist.var  # z

        self.masked_features = masked_features

        flow_list = [nn.Linear(in_features, hidden_features)]
        flow_list.append([nn.Linear(hidden_features,
                                    hidden_features)
                          for _ in range(num_layers-1)])
        flow_list.append([nn.Linear(hidden_features,
                                    2 * (in_features - masked_features))])
        self.flows = nn.ModuleList(flow_list)
        self.flow_name = "RealNVP"

        self.prob_text = "{}({} ; {})".format(
            self.flow_name,
            ','.join(var),
            dist.prob_text
        )
        self.prob_factorized_text = self.prob_text

    def _scale_translation(self, x):
        for i, flow in enumerate(self.flows[:-1]):
            x = F.relu(flow(x))
        x = self.flows[-1](x)
        scale, trans = torch.chunk(x, chunks=2, dim=-1)
        return scale, trans

    def forward(self, x, inverse=False, jacobian=False):
        x_0 = x[:, :self.masked_features]
        x_1 = x[:, self.masked_features:]
        scale, trans = self._scale_translation(x_0)

        if jacobian is False:
            if inverse:
                x_1 = (x_1 - trans) * torch.exp(-scale)
            else:
                x_1 = x_1 * torch.exp(scale) + trans
            output = torch.cat((x_0, x_1), dim=-1)

        else:
            output = torch.sum(scale, dim=-1)

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
        samples = self.forward(_x)
        # log p(z)
        log_dist = self.dist.log_likelihood({self.var_dist: samples})

        x_values = get_dict_values(x, self.var)
        logdet_jacobian = self.forward(x_values[0], jacobian=True)

        return log_dist + logdet_jacobian
