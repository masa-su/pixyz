from __future__ import print_function
import torch
from torch import nn
from torch.distributions import Normal as NormalTorch

from ..utils import tolist, get_dict_values
from .distributions import MultiplyDistribution


class NormalPoE(nn.Module):
    """
    p(z|x,y) \propto p(z)p(z|x)p(z|y)

    Parameters
    -------
    dists : list
    prior : Distribution

    Examples
    --------
    >>> poe = NormalPoE(c, [a, b])

    """

    def __init__(self, prior, dists=[], **kwargs):
        super(NormalPoE, self).__init__()

        self.prior = prior
        self.dists = nn.ModuleList(tolist(dists))
        var = self.prior.var

        cond_var = []
        for d in self.dists:
            if d.var != var:
                raise ValueError("Error")  # TODO: write the error message
            cond_var += d.cond_var

        self.cond_var = cond_var
        self.var = var

        if len(cond_var) == 0:
            self.prob_text = "p({})".format(
                ','.join(var)
            )
        else:
            self.prob_text = "p({}|{})".format(
                ','.join(var),
                ','.join(cond_var)
            )
        self.prob_factorized_text = self.prob_text

        self.distribution_name = "Normal"
        self.DistributionTorch = NormalTorch

    def _set_distribution(self, x={}, **kwargs):
        params = self.get_params(x, **kwargs)
        self.dist = self.DistributionTorch(**params)

    def _get_sample(self, reparam=True,
                    sample_shape=torch.Size()):

        if reparam:
            try:
                return self.dist.rsample(sample_shape=sample_shape)
            except NotImplementedError:
                print("We can not use the reparameterization trick"
                      "for this distribution.")

        return self.dist.sample(sample_shape=sample_shape)

    def get_params(self, params, **kwargs):
        loc = []
        scale = []

        if len(params) == 0:
            raise ValueError("You should set inputs or parameters")

        for d in self.dists:
            inputs = get_dict_values(params, d.cond_var, True)
            if len(inputs) != 0:
                outputs = d.get_params(inputs, **kwargs)
                loc.append(outputs["loc"])
                scale.append(outputs["scale"])

        outputs = self.prior.get_params({}, **kwargs)
        prior_loc = torch.ones_like(loc[0]).type(loc[0].dtype)
        prior_scale = torch.ones_like(scale[0]).type(scale[0].dtype)
        loc.append(outputs["loc"] * prior_loc)
        scale.append(outputs["scale"] * prior_scale)

        loc = torch.stack(loc)
        scale = torch.stack(scale)

        loc, scale = self.experts(loc, scale)

        return {"loc": loc, "scale": scale}

    def experts(self, loc, scale, eps=1e-8):
        T = 1. / (scale + eps)
        pd_loc = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
        pd_scale = 1. / torch.sum(T, dim=0) + eps
        return pd_loc, pd_scale

    def sample(self, x=None, return_all=True, **kwargs):
        # input : tensor, list or dict
        # output : dict

        self._set_distribution(x, **kwargs)
        output = {self.var[0]: self._get_sample(**kwargs)}

        if return_all:
            output.update(x)

        return output

    def log_likelihood(self, x):
        NotImplementedError

    def sample_mean(self, x, **kwargs):
        params = self.get_params(x, **kwargs)
        return params["loc"]

    def __mul__(self, other):
        return MultiplyDistribution(self, other)
