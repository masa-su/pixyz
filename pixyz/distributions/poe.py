from __future__ import print_function
import torch
from torch import nn

from ..utils import tolist, get_dict_values
from .exponential_distributions import Normal


class ProductOfNormal(Normal):
    r"""Product of normal distributions.

    .. math::
        p(z|x,y) \propto p(z)p(z|x)p(z|y)

    In this model, :math:`p(z|x)` and :math:`p(a|y)` perform as `experts` and :math:`p(z)` corresponds
    a prior of `experts`.

    See [Vedantam+ 2017] and [Wu+ 2017] for details.

    Examples
    --------
    >>> # pon = ProductOfNormal([p_x, p_y])
    >>> # pon.sample({"x": x, "y": y})
    >>> # pon.sample({"y": y})
    >>> # pon.sample()  # same as sampling from unit Gaussian.

    """

    def __init__(self, p=[], name="p", dim=1):
        """
        Parameters
        ----------
        p : :obj:`list` of :class:`pixyz.distributions.Normal`.
            List of experts.
        name : :obj:`str`, defaults to "p"
            Name of this distribution.
            This name is displayed in prob_text and prob_factorized_text.
        dim : :obj:`int`, defaults to 1
            Number of dimensions of this distribution.
            This might be ignored depending on the shape which is set in the sample method and on its parent distribution.
            Moreover, this is not consider when this class is inherited by DNNs.

        """
        p = tolist(p)
        if len(p) == 0:
            raise ValueError

        var = p[0].var
        cond_var = []

        for _p in p:
            if _p.var != var:
                raise ValueError

            if _p.distribution_name != "Normal":
                raise ValueError

            cond_var += _p.cond_var

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim)
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
        # experts
        if len(params_dict) > 0:
            loc, scale = self._get_expert_params(params_dict, **kwargs)  # (n_expert, n_batch, output_dim)
        else:
            loc = torch.zeros(1)
            scale = torch.zeros(1)

        output_loc, output_scale = self._compute_expert_params(loc, scale)
        output_dict = {"loc": output_loc, "scale": output_scale}

        return output_dict

    @staticmethod
    def _compute_expert_params(loc, scale):
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
        # parameter for prior
        prior_prec = 1  # prior_loc is not specified because it is equal to 0.

        # compute the diagonal precision matrix.
        prec = torch.zeros_like(scale).type(scale.dtype)
        prec[scale != 0] = 1. / scale[scale != 0]

        # compute the square root of a diagonal covariance matrix for the product of distributions.
        output_prec = torch.sum(prec, dim=0) + prior_prec
        output_variance = 1. / output_prec   # (n_batch, output_dim)

        # compute the mean vectors for the product of normal distributions.
        output_loc = torch.sum(prec * loc, dim=0)   # (n_batch, output_dim)
        output_loc = output_loc * output_variance

        return output_loc, torch.sqrt(output_variance)

    def _check_input(self, x, var=None):
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

        return checked_x

    def log_prob(self, sum_features=True, feature_dims=None):
        raise NotImplementedError

    def prob(self, sum_features=True, feature_dims=None):
        raise NotImplementedError

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        raise NotImplementedError


class ElementWiseProductOfNormal(ProductOfNormal):
    r"""Product of normal distributions.
    In this distribution, each element of the input vector on the given distribution is considered as
    a different expert.

    .. math::
        p(z|x) = p(z|x_1, x_2) \propto p(z)p(z|x_1)p(z|x_2)

    Examples
    --------
    >>> # pon = ElementWiseProductOfNormal(p)
    >>> # pon.sample({"x": x})
    >>> # pon.sample({"x": torch.zeros_like(x)})  # same as sampling from unit Gaussian.

    """

    def __init__(self, p, name="p", dim=1):
        """
        Parameters
        ----------
        p : pixyz.distributions.Normal
            Each element of this input vector is considered as a different expert.
            When some elements are 0, experts corresponding to these elements are considered not to be specified.
            :math:`p(z|x) = p(z|x_1, x_2=0) \propto p(z)p(z|x_1)`
        name : str, defaults to "p"
            Name of this distribution.
            This name is displayed in prob_text and prob_factorized_text.
        dim : int, defaults to 1
            Number of dimensions of this distribution.
            This might be ignored depending on the shape which is set in the sample method and on its parent
            distribution.
            Moreover, this is not consider when this class is inherited by DNNs.

        """
        if len(p.cond_var) != 1:
            raise ValueError

        super().__init__(p=p, name=name, dim=dim)

    def _check_input(self, x, var=None):
        if var is None:
            var = self.input_var

        if type(x) is torch.Tensor:
            checked_x = {var[0]: x}

        elif type(x) is list:
            # TODO: we need to check if all the elements contained in this list are torch.Tensor.
            checked_x = dict(zip(var, x))

        elif type(x) is dict:
            if not (set(list(x.keys())) >= set(var)):
                raise ValueError("Input keys are not valid.")
            checked_x = x

        else:
            raise ValueError("The type of input is not valid, got %s." % type(x))

        return checked_x

    @staticmethod
    def _get_mask(inputs, index):
        """Get a mask to the input to specify an expert identified by index.

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
        return mask

    def _get_params_with_masking(self, inputs, index, **kwargs):
        """Get the output parameters of the index-specified expert.

        Parameters
        ----------
        inputs : torch.Tensor
        index : int
        **kwargs
            Arbitrary keyword arguments.
        Returns
        -------
        outputs : torch.Tensor

        Examples
        --------
        >>> a = tensor([[1, 0, 0],
        ....            [0, 1, 0]])
        >>> self._get_params_with_masking(a, 0)
        tensor([[[0.01, 0.0131],
                 [0, 0]],  # loc
                [[0.42, 0.39],
                 [1, 1]],  # scale
               ])
        >>> self._get_params_with_masking(a, 1)
        tensor([[[0, 0],
                 [0.021, 0.11]],  # loc
                [[1, 1],
                 [0.293, 0.415]],  # scale
               ])
        >>> self._get_params_with_masking(a, 2)
        tensor([[[0, 0],
                 [0, 0]],  # loc
                [[1, 1],
                 [1, 1]],  # scale
               ])
        """
        mask = self._get_mask(inputs, index)  # (n_batch, n_expert)
        outputs_dict = self.p.get_params({self.cond_var[0]: inputs * mask}, **kwargs)
        outputs = torch.stack([outputs_dict["loc"], outputs_dict["scale"]])  # (2, n_batch, output_dim)

        # When the index-th expert in the output examples is not specified, set zero to them.
        outputs[:, inputs[:, index] == 0, :] = 0
        return outputs

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

        outputs = [self._get_params_with_masking(inputs, i) for i in range(n_expert)]
        outputs = torch.stack(outputs)  # (n_expert, 2, n_batch, output_dim)

        return outputs[:, 0, :, :], outputs[:, 1, :, :]  # (n_expert, n_batch, output_dim)
