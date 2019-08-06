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

    References
    ----------
    [Vedantam+ 2017] Generative Models of Visually Grounded Imagination

    [Wu+ 2018] Multimodal Generative Models for Scalable Weakly-Supervised Learning

    Examples
    --------
    >>> pon = ProductOfNormal([p_x, p_y]) # doctest: +SKIP
    >>> pon.sample({"x": x, "y": y}) # doctest: +SKIP
    {'x': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]],),
     'y': tensor([[0., 0., 0.,  ..., 0., 0., 1.],
         [0., 0., 1.,  ..., 0., 0., 0.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 1., 0.],
         [1., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.]]),
     'z': tensor([[ 0.6611,  0.3811,  0.7778,  ..., -0.0468, -0.3615, -0.6569],
         [-0.0071, -0.9178,  0.6620,  ..., -0.1472,  0.6023,  0.5903],
         [-0.3723, -0.7758,  0.0195,  ...,  0.8239, -0.3537,  0.3854],
         ...,
         [ 0.7820, -0.4761,  0.1804,  ..., -0.5701, -0.0714, -0.5485],
         [-0.1873, -0.2105, -0.1861,  ..., -0.5372,  0.0752,  0.2777],
         [-0.2563, -0.0828,  0.1605,  ...,  0.2767, -0.8456,  0.7364]])}
    >>> pon.sample({"y": y}) # doctest: +SKIP
    {'y': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.],
         [0., 0., 0.,  ..., 1., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]),
     'z': tensor([[-0.3264, -0.4448,  0.3610,  ..., -0.7378,  0.3002,  0.4370],
         [ 0.0928, -0.1830,  1.1768,  ...,  1.1808, -0.7226, -0.4152],
         [ 0.6999,  0.2222, -0.2901,  ...,  0.5706,  0.7091,  0.5179],
         ...,
         [ 0.5688, -1.6612, -0.0713,  ..., -0.1400, -0.3903,  0.2533],
         [ 0.5412, -0.0289,  0.6365,  ...,  0.7407,  0.7838,  0.9218],
         [ 0.0299,  0.5148, -0.1001,  ...,  0.9938,  1.0689, -1.1902]])}
    >>> pon.sample()  # same as sampling from unit Gaussian. # doctest: +SKIP
    {'z': tensor(-0.4494)}

    """

    def __init__(self, p=[], name="p", features_shape=torch.Size()):
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

        var = p[0].var
        cond_var = []

        for _p in p:
            if _p.var != var:
                raise ValueError()

            if _p.distribution_name != "Normal":
                raise ValueError()

            cond_var += _p.cond_var

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape)
        if len(p) == 1:
            self.p = p[0]
        else:
            self.p = nn.ModuleList(p)

    @property
    def prob_factorized_text(self):
        prob_text = "p({})".format(
            ','.join(self._var)
        )

        if len(self._cond_var) != 0:
            prob_text += "".join([p.prob_text for p in self.p])

        return prob_text

    @property
    def prob_joint_factorized_and_text(self):
        """str: Return a formula of the factorized probability distribution."""
        if self.prob_factorized_text == self.prob_text:
            prob_text = self.prob_text
        else:
            prob_text = "{} \\propto {}".format(self.prob_text, self.prob_factorized_text)
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
        raise NotImplementedError()

    def prob(self, sum_features=True, feature_dims=None):
        raise NotImplementedError()

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        raise NotImplementedError()


class ElementWiseProductOfNormal(ProductOfNormal):
    r"""Product of normal distributions.
    In this distribution, each element of the input vector on the given distribution is considered as
    a different expert.

    .. math::
        p(z|x) = p(z|x_1, x_2) \propto p(z)p(z|x_1)p(z|x_2)

    Examples
    --------
    >>> pon = ElementWiseProductOfNormal(p) # doctest: +SKIP
    >>> pon.sample({"x": x}) # doctest: +SKIP
    {'x': tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]),
     'z': tensor([[-0.3572, -0.0632,  0.4872,  0.2269, -0.1693, -0.0160, -0.0429,  0.2017,
          -0.1589, -0.3380, -0.9598,  0.6216, -0.4296, -1.1349,  0.0901,  0.3994,
           0.2313, -0.5227, -0.7973,  0.3968,  0.7137, -0.5639, -0.4891, -0.1249,
           0.8256,  0.1463,  0.0801, -1.2202,  0.6984, -0.4036,  0.4960, -0.4376,
           0.3310, -0.2243, -0.2381, -0.2200,  0.8969,  0.2674,  0.4681,  1.6764,
           0.8127,  0.2722, -0.2048,  0.1903, -0.1398,  0.0099,  0.4382, -0.8016,
           0.9947,  0.7556, -0.2017, -0.3920,  1.4212, -1.2529, -0.1002, -0.0031,
           0.1876,  0.4267,  0.3622,  0.2648,  0.4752,  0.0843, -0.3065, -0.4922],
         [ 0.3770, -0.0413,  0.9102,  0.2897, -0.0567,  0.5211,  1.5233, -0.3539,
           0.5163, -0.2271, -0.1027,  0.0294, -1.4617,  0.1640,  0.2025, -0.2190,
           0.0555,  0.5779, -0.2930, -0.2161,  0.2835, -0.0354, -0.2569, -0.7171,
           0.0164, -0.4080,  1.1088,  0.3947,  0.2720, -0.0600, -0.9295, -0.0234,
           0.5624,  0.4866,  0.5285,  1.1827,  0.2494,  0.0777,  0.7585,  0.5127,
           0.7500, -0.3253,  0.0250,  0.0888,  1.0340, -0.1405, -0.8114,  0.4492,
           0.2725, -0.0270,  0.6379, -0.8096,  0.4259,  0.3179, -0.1681,  0.3365,
           0.6305,  0.5203,  0.2384,  0.0572,  0.4804,  0.9553, -0.3244,  1.5373]])}
    >>> pon.sample({"x": torch.zeros_like(x)})  # same as sampling from unit Gaussian. # doctest: +SKIP
    {'x': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
     'z': tensor([[-0.7777, -0.5908, -1.5498, -0.7505,  0.6201,  0.7218,  1.0045,  0.8923,
          -0.8030, -0.3569,  0.2932,  0.2122,  0.1640,  0.7893, -0.3500, -1.0537,
          -1.2769,  0.6122, -1.0083, -0.2915, -0.1928, -0.7486,  0.2418, -1.9013,
           1.2514,  1.3035, -0.3029, -0.3098, -0.5415,  1.1970, -0.4443,  2.2393,
          -0.6980,  0.2820,  1.6972,  0.6322,  0.4308,  0.8953,  0.7248,  0.4440,
           2.2770,  1.7791,  0.7563, -1.1781, -0.8331,  0.1825,  1.5447,  0.1385,
          -1.1348,  0.0257,  0.3374,  0.5889,  1.1231, -1.2476, -0.3801, -1.4404,
          -1.3066, -1.2653,  0.5958, -1.7423,  0.7189, -0.7236,  0.2330,  0.3117],
         [ 0.5495,  0.7210, -0.4708, -2.0631, -0.6170,  0.2436, -0.0133, -0.4616,
          -0.8091, -0.1592,  1.3117,  0.0276,  0.6625, -0.3748, -0.5049,  1.8260,
          -0.3631,  1.1546, -1.0913,  0.2712,  1.5493,  1.4294, -2.1245, -2.0422,
           0.4976, -1.2785,  0.5028,  1.4240,  1.1983,  0.2468,  1.1682, -0.6725,
          -1.1198, -1.4942, -0.3629,  0.1325, -0.2256,  0.4280,  0.9830, -1.9427,
          -0.2181,  1.1850, -0.7514, -0.8172,  2.1031, -0.1698, -0.3777, -0.7863,
           1.0936, -1.3720,  0.9999,  1.3302, -0.8954, -0.5999,  2.3305,  0.5702,
          -1.0767, -0.2750, -0.3741, -0.7026, -1.5408,  0.0667,  1.2550, -0.5117]])}

    """

    def __init__(self, p, name="p", features_shape=torch.Size()):
        r"""
        Parameters
        ----------
        p : pixyz.distributions.Normal
            Each element of this input vector is considered as a different expert.
            When some elements are 0, experts corresponding to these elements are considered not to be specified.
            :math:`p(z|x) = p(z|x_1, x_2=0) \propto p(z)p(z|x_1)`
        name : str, defaults to "p"
            Name of this distribution.
            This name is displayed in prob_text and prob_factorized_text.
        features_shape : :obj:`torch.Size` or :obj:`list`, defaults to torch.Size())
            Shape of dimensions (features) of this distribution.

        """
        if len(p.cond_var) != 1:
            raise ValueError()

        super().__init__(p=p, name=name, features_shape=features_shape)

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
        >>> # pon = ElementWiseProductOfNormal(p)
        >>> # a = torch.tensor([[1, 0, 0], [0, 1, 0]])
        >>> # pon._get_params_with_masking(a, 0)
        tensor([[[0.01, 0.0131],
                 [0, 0]],  # loc
                [[0.42, 0.39],
                 [1, 1]],  # scale
               ])
        >>> # pon._get_params_with_masking(a, 1)
        tensor([[[0, 0],
                 [0.021, 0.11]],  # loc
                [[1, 1],
                 [0.293, 0.415]],  # scale
               ])
        >>> # self._get_params_with_masking(a, 2)
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
