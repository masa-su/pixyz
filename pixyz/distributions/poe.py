from __future__ import print_function
import torch
from torch import nn

from ..utils import tolist, get_dict_values
from .exponential_distributions import Normal


class ProductOfNormal(Normal):
    """Product of normal distributions.

    :math:`p(z|x,y) \propto p(z)p(z|x)p(z|y)`

    In this model, p(z|x) and p(a|y) perform as `experts` and p(z) corresponds a prior of `experts`.

    See [Vedantam+ 2017] and [Wu+ 2017] for details.

    Attributes
    ----------
    p : list of Normal
        List of experts.
    prior : Normal
        A prior of experts.
    name : str, default "p"
        Name of this distribution.
        This name is displayed in prob_text and prob_factorized_text.
    dim : int, default 1
        Number of dimensions of this distribution.
        This might be ignored depending on the shape which is set in the sample method and on its parent distribution.
        Moreover, this is not consider when this class is inherited by DNNs.

    Examples
    --------
    >>> pon = ProductOfNormal(prior, [p_x, p_y])

    """

    def __init__(self, prior, p=[], name="p", dim=1):
        if prior.distribution_name != "Normal":
            raise ValueError
        var = prior.var

        cond_var = []
        p = tolist(p)
        for _p in p:
            if _p.var != var:
                raise ValueError

            if _p.distribution_name != "Normal":
                raise ValueError

            cond_var += _p.cond_var

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim)
        self.prior = prior
        if len(p) == 1:
            self.p = p[0]
        else:
            self.p = nn.ModuleList(p)

    @property
    def prob_factorized_text(self):
        if len(self._cond_var) == 0:
            prob_text = "p({})".format(
                ','.join(self._var)
            )
        else:
            prob_text = "p({}|{})".format(
                ','.join(self._var),
                ','.join(self._cond_var)
            )

        return prob_text

    def _get_expert_params(self, params_dict={}, **kwargs):
        """Get the output parameters of all experts.

        Parameters
        ----------
        params_dict : dict
        **kwargs
            Arbitrary keyword arguments.
        Returns
        -------
        torch.Tensor

        """

        loc = []
        scale = []

        for _p in self.p:
            inputs_dict = get_dict_values(params_dict, _p.cond_var, True)
            if len(inputs_dict) != 0:
                outputs = _p.get_params(inputs_dict, **kwargs)
                loc.append(outputs["loc"])
                scale.append(outputs["scale"])

        loc = torch.stack(loc)
        scale = torch.stack(scale)

        return loc, scale

    def get_params(self, params_dict={}, **kwargs):
        loc_all = []
        scale_all = []

        # experts
        if len(params_dict) > 0:
            loc, scale = self._get_expert_params(params_dict, **kwargs)  # (n_expert, n_batch, output_dim)
        else:
            loc = torch.zeros(1)
            scale = torch.zeros(1)

        loc_all.append(loc)
        scale_all.append(scale)

        # prior
        outputs = self.prior.get_params({}, **kwargs)
        prior_loc = torch.ones_like(loc[:1]).type(loc[:1].dtype)
        prior_scale = torch.ones_like(scale[:1]).type(scale[:1].dtype)
        prior_loc = prior_loc * outputs["loc"]
        prior_scale = prior_scale * outputs["scale"]
        loc_all.append(prior_loc)
        scale_all.append(prior_scale)

        loc_all = torch.cat(loc_all, dim=0)
        scale_all = torch.cat(scale_all, dim=0)

        output_loc, output_scale = self._compute_expert_params(loc_all, scale_all)
        output_dict = {"loc": output_loc, "scale": output_scale}

        return output_dict

    @staticmethod
    def _compute_expert_params(loc, scale, eps=1e-8):
        """Compute parameters for the product of experts.
        Is is assumed that unspecified experts are excluded from inputs.

        Parameters
        ----------
        loc : torch.Tensor (n_expert, n_batch, output_dim)
            Concatenation of mean vectors for specified experts.

        scale : torch.Tensor (n_expert, n_batch, output_dim)
            Concatenation of the square root of a diagonal covariance matrix for specified experts.

        eps : float, default 1e-8
            A constant value for avoiding division by 0.

        Returns
        -------
        output_loc : torch.Tensor (n_batch, output_dim)
            Mean vectors for this distribution.

        output_scale : torch.Tensor (n_batch, output_dim)
            The square root of diagonal covariance matrices for this distribution.
        """

        # compute the diagonal precision matrix.
        prec = 1. / (scale**2 + eps)

        # compute the square root of a diagonal covariance matrix for the product of distributions.
        output_prec = torch.sum(prec, dim=0)
        output_scale = torch.sqrt(1. / output_prec)

        # compute the mean vectors for the product of normal distributions.
        output_loc = torch.sum(prec * loc, dim=0) * output_scale**2
        output_loc = output_loc

        return output_loc, output_scale

    def _check_input(self, x, var=None):
        if var is None:
            var = self.input_var

        if type(x) is torch.Tensor:
            checked_x = {var[0]: x}

        elif type(x) is list:
            # TODO: we need to check if all the elements contained in this list are torch.Tensor.
            checked_x = dict(zip(var, x))

        elif type(x) is dict:
            if not (set(list(x.keys())) <= set(var)):  # point of modification
                raise ValueError("Input keys are not valid.")
            checked_x = x

        else:
            raise ValueError("The type of input is not valid, got %s." % type(x))

        return checked_x

    def log_prob(self, sum_features=True, feature_dims=None):
        raise NotImplementedError

    def prob(self, sum_features=True, feature_dims=None):
        raise NotImplementedError

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        raise NotImplementedError


class ElementWiseProductOfNormal(ProductOfNormal):
    """Product of normal distributions.
    In this distribution, each element of the input vector on the given distribution is considered as
     a different expert.

    :math:`p(z|x) = p(z|x_1, x_2) \propto p(z)p(z|x_1)p(z|x_2)`

    Attributes
    ----------
    p : Normal
        Each element of this input vector is considered as a different expert.
        When some elements are 0, experts corresponding to these elements are considered not to be specified.

        :math:`p(z|x) = p(z|x_1, 0) \propto p(z)p(z|x_1)`

    prior : Normal
        A prior of experts.
    name : str, default "p"
        Name of this distribution.
        This name is displayed in prob_text and prob_factorized_text.
    dim : int, default 1
        Number of dimensions of this distribution.
        This might be ignored depending on the shape which is set in the sample method and on its parent distribution.
        Moreover, this is not consider when this class is inherited by DNNs.

    Examples
    --------
    >>> pon = ElementWiseProductOfNormal(prior, p)

    """

    def __init__(self, prior, p, name="p", dim=1):
        if len(p.cond_var) != 1:
            raise ValueError

        super().__init__(prior=prior, p=p, name=name, dim=dim)

    @staticmethod
    def _masking(inputs, index):
        """Apply a mask to the input to specify experts identified by index.

        Parameters
        ----------
        inputs : torch.Tensor
        index : int

        Returns
        -------
        torch.Tensor

        """
        mask = torch.zeros_like(inputs).type(inputs.dtype)
        mask[:, index] = 1
        return inputs * mask

    def _get_params_with_masking(self, inputs, index, **kwargs):
        """Get the output parameters of experts specified by index.

        Parameters
        ----------
        inputs : torch.Tensor
        index : int
        **kwargs
            Arbitrary keyword arguments.
        Returns
        -------
        torch.Tensor

        """
        outputs = self.p.get_params({self.cond_var: self._masking(inputs, index)}, **kwargs)
        return torch.stack([outputs["loc"], outputs["scale"]])  # (n_batch, output_dim, 2)

    def _get_expert_params(self, params_dict={}, **kwargs):
        """Get the output parameters of all experts.

        Parameters
        ----------
        params_dict : dict
        **kwargs
            Arbitrary keyword arguments.
        Returns
        -------
        torch.Tensor
        torch.Tensor

        """
        inputs = get_dict_values(params_dict, self.cond_var)[0]  # (n_batch, n_expert=input_dim)

        n_expert = inputs.size()[1]

        outputs = [self._get_params_with_masking(inputs, i) for i in n_expert]
        outputs = torch.stack(outputs)  # (n_batch, output_dim, 2, n_expert)

        return outputs[:, :, 0, :], outputs[:, :, 1, :]
