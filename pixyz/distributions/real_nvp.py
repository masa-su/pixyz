from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_dict_values, epsilon
from .distributions import Distribution


class RealNVP(Distribution):
    def __init__(self, prior, dim,
                 num_multiscale_layers=2,
                 var=[], image=False, name="p",
                 **kwargs):
        super(RealNVP, self).__init__(cond_var=prior.cond_var, var=var, name=name, dim=dim)
        self.prior = prior
        self.var = var  # x
        self.cond_var = self.dist.cond_var
        self.var_dist = self.dist.var  # z

        flow_list = [MultiScaleLayer1D(dim, layer_id=layer_id, **kwargs)
                     for layer_id in range(num_multiscale_layers)]

        self.image = image

        self.flows = nn.ModuleList(flow_list)
        self._flow_name = "RealNVP"

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

    def forward(self, x, inverse=False, jacobian=False):
        logdet_jacobian = 0

        if inverse is False:
            _flows = self.flows
            x_use = x
            x_disuse = None

            if self.image:
                # corrupt data (Tapani Raiko's dequantization)
                x_use = x_use * 255.0
                corruption_level = 1.0
                x_use = x_use +\
                    corruption_level * torch.empty_like(x_use).uniform_(0, 1)
                x_use = x_use / (255.0 + corruption_level)

                # model logit
                alpha = .05
                # avoid 0 when x_use = 1
                x_use = x_use * (1 - alpha) + alpha * epsilon()
                jac = torch.sum(-torch.log(x_use)
                                - torch.log(1 - x_use), dim=1)
                x_use = torch.log(x_use) - torch.log(1 - x_use)
                logdet_jacobian += jac

        else:
            _flows = self.flows[::-1]
            x_use = None
            x_disuse = x

        if jacobian is False:
            for i, flow in enumerate(_flows):
                x_use, x_disuse = flow(x_use, x_disuse, inverse=inverse)

        else:
            for i, flow in enumerate(_flows):
                x_use, x_disuse, _logdet_jacobian = flow(x_use, x_disuse,
                                                         jacobian=jacobian,
                                                         inverse=inverse)
                logdet_jacobian += _logdet_jacobian

        if inverse is False:
            x = torch.cat((x_disuse, x_use), dim=1)

        else:
            x = x_use
            if self.image:
                # inverse logit
                x = 1. / (1 + torch.exp(-x))

        if jacobian is False:
            return x

        else:
            return x, logdet_jacobian

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

        samples[self.var_dist[0]] = output
        return samples

    def log_likelihood(self, x):
        # use a bijection function
        # z=f(x)
        _x = get_dict_values(x, self.var)
        z, logdet_jacobian = self.forward(_x[0], jacobian=True)

        # log p(z)
        log_dist = self.dist.log_likelihood({self.var_dist[0]: z})

        output = log_dist + logdet_jacobian

        """
        if self.image:
            output -= _x[0].shape[1]
             * torch.log(torch.tensor(256.).to(_x[0].device))
        """

        return output


class MultiScaleLayer1D(nn.Module):
    def __init__(self, in_features, layer_id,
                 hidden_features=64,
                 num_nn_layers=2,
                 num_flow_layers=3):
        super(MultiScaleLayer1D, self).__init__()

        self.in_features = in_features // (2 ** layer_id)
        flow_list =\
            [AffineCouplingLayer1D(self.in_features,
                                   hidden_features=hidden_features,
                                   num_layers=num_nn_layers,
                                   pattern=i % 2)
             for i in range(num_flow_layers)]

        self.flows = nn.ModuleList(flow_list)
        self.split = SplitLayer(layer_id)

    def forward(self, x_use, x_disuse, inverse=False, jacobian=False):
        if inverse is False:
            _flows = self.flows
        else:
            _flows = self.flows[::-1]

        if inverse:
            x_use, x_disuse = self.split.forward(x_use, x_disuse,
                                                 inverse=inverse)

        if jacobian is False:
            for i, flow in enumerate(_flows):
                x_use = flow(x_use, inverse=inverse)

        else:
            logdet_jacobian = 0
            for i, flow in enumerate(_flows):
                x_use, _logdet_jacobian = flow(x_use, jacobian=jacobian,
                                               inverse=inverse)
                logdet_jacobian += _logdet_jacobian

        if inverse is False:
            x_use, x_disuse = self.split.forward(x_use, x_disuse,
                                                 inverse=inverse)

        if jacobian is False:
            return x_use, x_disuse

        else:
            return x_use, x_disuse, logdet_jacobian


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features,
                 masked_type="checkerboard",
                 pattern=0  # 0 or 1
                 ):
        super(AffineCouplingLayer, self).__init__()

        self.in_features = in_features
        self.masked_type = masked_type
        self.pattern = pattern

    def _scale_translation(self, x):
        NotImplementedError

    def _masking(self, x, reverse=False):
        NotImplementedError

    def forward(self, x, inverse=False, jacobian=False):
        # forward: (x->z)
        # inverse: (z->x)

        x_0 = self._masking(x, False)
        x_1 = self._masking(x, True)
        scale, trans = self._scale_translation(x_0)

        if inverse:
            x_1 = (x_1 - trans) / torch.exp(scale)
        else:
            x_1 = x_1 * torch.exp(scale) + trans

        output = x_0 + x_1

        if jacobian:
            logdet_jacobian = torch.sum(scale, dim=1)  # 1D
            return output, logdet_jacobian

        return output

    def extra_repr(self):
        return 'in_features={}, pattern={}'.format(self.in_features,
                                                   self.pattern)


class AffineCouplingLayer1D(AffineCouplingLayer):
    def __init__(self, in_features,
                 hidden_features=512,
                 num_layers=2,
                 masked_type="checkerboard",
                 pattern=0  # 0 or 1
                 ):
        super(AffineCouplingLayer1D,
              self).__init__(in_features,
                             masked_type=masked_type,
                             pattern=pattern)

        self.hidden_features = hidden_features

        layer_list = [nn.Linear(in_features,
                                hidden_features)]
        layer_list += [nn.Linear(hidden_features,
                                 hidden_features)
                       for _ in range(num_layers-2)]
        layer_list += [nn.Linear(hidden_features,
                                 2 * in_features)]

        self.layers = nn.ModuleList(layer_list)

        batch_norms = [nn.BatchNorm1d(hidden_features)
                       for _ in range(num_layers-1)]
        self.batch_norms = nn.ModuleList(batch_norms)

    def _scale_translation(self, x):
        for layer, batch_norm in zip(self.layers[:-1],
                                     self.batch_norms):
            x = F.relu(batch_norm(layer(x)))
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        """

        x = self.layers[-1](x)
        scale, trans = torch.chunk(x, chunks=2, dim=-1)

        scale = self._masking(torch.tanh(scale), True)
        trans = self._masking(trans, True)
        return scale, trans

    def _masking(self, x, reverse=False):
        x_shape = x.shape
        if self.masked_type == "checkerboard":
            mask = torch.zeros(x_shape[1]).to(x.device)
            mask[self.pattern::2] = 1
        else:
            NotImplementedError

        if reverse:
            return x * (1 - mask).unsqueeze(0)
        else:
            return x * mask.unsqueeze(0)


class AffineCouplingLayer2D(AffineCouplingLayer):
    def __init__(self, in_features,
                 hidden_features=512,
                 num_layers=2,
                 masked_type="checkerboard",
                 pattern=0  # 0 or 1
                 ):
        super(AffineCouplingLayer2D,
              self).__init__(in_features,
                             masked_type=masked_type,
                             pattern=pattern)

        self.hidden_features = hidden_features

        flow_list = [nn.Linear(in_features,
                               hidden_features)]
        flow_list += [nn.Linear(hidden_features,
                                hidden_features)
                      for _ in range(num_layers-1)]
        flow_list += [nn.Linear(hidden_features,
                                2 * in_features)]

        self.flows = nn.ModuleList(flow_list)

    def _scale_translation(self, x):
        for flow in self.flows[:-1]:
            x = F.relu(flow(x))
        x = self.flows[-1](x)
        scale, trans = torch.chunk(x, chunks=2, dim=-1)

        scale = self._masking(scale, True)
        trans = self._masking(trans, True)
        return scale, trans

    def _masking(self, x, reverse=False):
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


class SplitLayer(object):
    #  Factorizing out/in layer
    def __init__(self, layer_id):
        self.layer_id = layer_id

    def get_split(self, x, split):
        x_shape = x.shape
        assert x_shape != 2 and x_shape != 4, \
            NotImplementedError

        if len(x.shape) == 2:
            return x[:, :split], x[:, split:]
        else:
            return x[:, :, :, :split], x[:, :, :, split:]

    def forward(self, x_use, x_disuse, inverse=False):
        if inverse is False:
            # increase x_disuse, decrease x_use
            x_use_shape = x_use.shape
            assert x_use_shape != 2 and x_use_shape != 4, \
                NotImplementedError
            split_dim = len(x_use_shape) - 1

            split = x_use_shape[split_dim] // 2
            _x_disuse, x_use = self.get_split(x_use, split)

            if x_disuse is not None:
                x_disuse = torch.cat((x_disuse, _x_disuse),
                                     dim=split_dim)
            else:
                x_disuse = _x_disuse

        else:
            # increase x_use, decrease x_disuse
            x_disuse_shape = x_disuse.shape
            assert x_disuse_shape != 2 and x_disuse_shape != 4, \
                NotImplementedError
            split_dim = len(x_disuse_shape) - 1

            if x_use is not None:
                split = x_use.shape[split_dim]
            else:
                split = x_disuse_shape[split_dim] // (2 ** self.layer_id)

            x_disuse, _x_use = self.get_split(x_disuse, -split)

            if x_use is not None:
                x_use = torch.cat((_x_use, x_use), dim=split_dim)
            else:
                x_use = _x_use

        return x_use, x_disuse
