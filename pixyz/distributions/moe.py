from __future__ import print_function
import torch
from torch import nn
import numpy as np

from ..utils import tolist, get_dict_values
from ..distributions import Normal


class MixtureOfNormal(Normal):
    r"""Mixture of normal distributions.
    .. math::
        p(z|x,y) = p(z|x) + p(z|y)
    In this models, :math:`p(z|x)` and :math:`p(a|y)` perform as `experts`.

    References
    ----------
    [Shi+ 2019] Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models

    """

    def __init__(self, p=[], weight_modalities=None, name="p", features_shape=torch.Size()):
        """
        Parameters
        ----------
        p : :obj:`list` of :class:`pixyz.distributions.Normal`.
            List of experts.
        name : :obj:`str`, defaults to "p"
            Name of this distribution.
            This name is displayed in prob_text and prob_factorized_text.
        features_shape : :obj:`torch.Size` or :obj:`list`, defaults to torch.Size())
            Shape of dimensions (features) of this distribution.
        """

        p = tolist(p)
        if len(p) == 0:
            raise ValueError()

        if weight_modalities is None:
            weight_modalities = torch.ones(len(p)) / float(len(p))

        elif len(weight_modalities) != len(p):
            raise ValueError()

        var = p[0].var
        cond_var = []

        for _p in p:
            if _p.var != var:
                raise ValueError()

            cond_var += _p.cond_var

        cond_var = list(set(cond_var))

        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape)
        self.p = nn.ModuleList(p)
        self.weight_modalities = weight_modalities

    def _get_expert_params(self, params_dict={}, **kwargs):
        """Get the output parameters of all experts.
        Parameters
        ----------
        params_dict : dict
        **kwargs
            Arbitrary keyword arguments.
        Returns
        -------
        loc : torch.Tensor
            Concatenation of mean vectors for specified experts. (n_expert, n_batch, output_dim)
        scale : torch.Tensor
            Concatenation of the square root of a diagonal covariance matrix for specified experts.
            (n_expert, n_batch, output_dim)
        weight : np.array
            (n_expert, )
        """

        loc = []
        scale = []

        for i, _p in enumerate(self.p):
            inputs_dict = get_dict_values(params_dict, _p.cond_var, True)
            if len(inputs_dict) != 0:
                outputs = _p.get_params(inputs_dict, **kwargs)
                loc.append(outputs["loc"])
                scale.append(outputs["scale"])

        loc = torch.stack(loc)
        scale = torch.stack(scale)

        return loc, scale

    def get_params(self, params_dict={}, **kwargs):
        # experts
        if len(params_dict) > 0:
            loc, scale = self._get_expert_params(params_dict, **kwargs)  # (n_expert, n_batch, output_dim)
        else:
            raise ValueError()

        output_loc, output_scale = self._compute_expert_params(loc, scale)
        output_dict = {"loc": output_loc, "scale": output_scale}

        return output_dict

    def _compute_expert_params(self, loc, scale):
        """Compute parameters for the product of experts.
        Is is assumed that unspecified experts are excluded from inputs.
        Parameters
        ----------
        loc : torch.Tensor
            Concatenation of mean vectors for specified experts. (n_expert, n_batch, output_dim)
        scale : torch.Tensor
            Concatenation of the square root of a diagonal covariance matrix for specified experts.
            (n_expert, n_batch, output_dim)
        Returns
        -------
        output_loc : torch.Tensor
            Mean vectors for this distribution. (n_batch, output_dim)
        output_scale : torch.Tensor
            The square root of diagonal covariance matrices for this distribution. (n_batch, output_dim)
        """
        num_samples = loc.shape[1]

        idx_start = []
        idx_end = []
        for k in range(0, len(self.weight_modalities)):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k - 1])
            if k == len(self.weight_modalities) - 1:
                i_end = num_samples
            else:
                i_end = i_start + int(np.floor(num_samples * self.weight_modalities[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)

        idx_end[-1] = num_samples

        output_loc = torch.cat([loc[k, idx_start[k]:idx_end[k], :] for k in range(len(self.weight_modalities))])
        output_scale = torch.cat([scale[k, idx_start[k]:idx_end[k], :] for k in range(len(self.weight_modalities))])

        return output_loc, output_scale

    def _get_input_dict(self, x, var=None):
        if var is None:
            var = self.input_var

        if type(x) is torch.Tensor:
            checked_x = {var[0]: x}

        elif type(x) is list:
            # TODO: we need to check if all the elements contained in this list are torch.Tensor.
            checked_x = dict(zip(var, x))

        elif type(x) is dict:
            # point of modification
            checked_x = x

        else:
            raise ValueError("The type of input is not valid, got %s." % type(x))

        return get_dict_values(checked_x, var, return_dict=True)

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        log_prob = torch.stack([w * p.get_log_prob(x_dict, sum_features=sum_features, feature_dims=feature_dims) for p, w in zip(self.p, self.weight_modalities)])
        log_prob = torch.logsumexp(log_prob, dim=0)

        return log_prob
