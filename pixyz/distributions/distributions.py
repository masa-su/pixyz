from __future__ import print_function
import torch
import re
from torch import nn
from copy import deepcopy
from itertools import chain

from ..utils import tolist, convert_latex_name
from .sample_dict import SampleDict
from ..losses import LogProb, Prob


class Distribution(nn.Module):
    """Distribution class. In Pixyz, all distributions are required to inherit this class.


    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    >>> # Marginal distribution
    >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...             features_shape=[64], name="p1")
    >>> print(p1)
    Distribution:
      p_{1}(x)
    Network architecture:
      Normal(
        name=p_{1}, distribution_name=Normal,
        var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([64])
        (loc): torch.Size([1, 64])
        (scale): torch.Size([1, 64])
      )

    >>> # Conditional distribution
    >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
    ...             features_shape=[64], name="p2")
    >>> print(p2)
    Distribution:
      p_{2}(x|y)
    Network architecture:
      Normal(
        name=p_{2}, distribution_name=Normal,
        var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([64])
        (scale): torch.Size([1, 64])
      )

    >>> # Conditional distribution (by neural networks)
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["x"], cond_var=["y"], name="p3")
    ...         self.model_loc = nn.Linear(128, 64)
    ...         self.model_scale = nn.Linear(128, 64)
    ...     def forward(self, y):
    ...         return {"loc": self.model_loc(y), "scale": F.softplus(self.model_scale(y))}
    >>> p3 = P()
    >>> print(p3)
    Distribution:
      p_{3}(x|y)
    Network architecture:
      P(
        name=p_{3}, distribution_name=Normal,
        var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([])
        (model_loc): Linear(in_features=128, out_features=64, bias=True)
        (model_scale): Linear(in_features=128, out_features=64, bias=True)
      )
    """

    def __init__(self, var, cond_var=(), name="p"):
        """
        Parameters
        ----------
        var : :obj:`list` of :obj:`str`
            Variables of this distribution.
        cond_var : :obj:`list` of :obj:`str`, defaults to []
            Conditional variables of this distribution.
            In case that cond_var is not empty, we must set the corresponding inputs to sample variables.
        name : :obj:`str`, defaults to "p"
            Name of this distribution.
            This name is displayed in :attr:`prob_text` and :attr:`prob_factorized_text`.
        features_shape_TODO : :obj:`torch.Size` or :obj:`list`, defaults to torch.Size())
            Shape of dimensions (features) of this distribution.

        """
        super().__init__()

        cond_var = list(cond_var)
        var = list(var)
        _vars = cond_var + var
        if len(_vars) != len(set(_vars)):
            raise ValueError("There are conflicted variables.")

        self._cond_var = cond_var
        self._var = var

        self._name = convert_latex_name(name)

        self._prob_text = None
        self._prob_factorized_text = None

    @property
    def distribution_name(self):
        """str: Name of this distribution class."""
        return ""

    @property
    def name(self):
        """str: Name of this distribution displayed in :obj:`prob_text` and :obj:`prob_factorized_text`."""
        return self._name

    @name.setter
    def name(self, name):
        if type(name) is str:
            self._name = name
            return

        raise ValueError("Name of the distribution class must be a string type.")

    @property
    def var(self):
        """list: Variables of this distribution."""
        return self._var

    @property
    def cond_var(self):
        """list: Conditional variables of this distribution."""
        return self._cond_var

    @property
    def input_var(self):
        """list: Input variables of this distribution.
        Normally, it has same values as :attr:`cond_var`.

        """
        return self._cond_var

    @property
    def prob_text(self):
        """str: Return a formula of the (joint) probability distribution."""
        _var_text = [','.join([convert_latex_name(var_name) for var_name in self._var])]
        if len(self._cond_var) != 0:
            _var_text += [','.join([convert_latex_name(var_name) for var_name in self._cond_var])]

        _prob_text = "{}({})".format(
            self._name,
            "|".join(_var_text)
        )

        return _prob_text

    @property
    def prob_factorized_text(self):
        """str: Return a formula of the factorized probability distribution."""
        return self.prob_text

    @property
    def prob_joint_factorized_and_text(self):
        """str: Return a formula of the factorized and the (joint) probability distributions."""
        if self.prob_factorized_text == self.prob_text:
            prob_text = self.prob_text
        else:
            prob_text = "{} = {}".format(self.prob_text, self.prob_factorized_text)
        return prob_text

    # TODO: -多分廃止する
    def _check_input(self, input_, var=None):
        """Check the type of given input.
        If the input type is :obj:`dict`, this method checks whether the input keys contains the :attr:`var` list.
        In case that its type is :obj:`list` or :obj:`tensor`, it returns the output formatted in :obj:`dict`.

        Parameters
        ----------
        input_ : :obj:`torch.Tensor`, :obj:`list`, or :obj:`dict`, or :obj:`SampleDict`
            Input variables.
        var : :obj:`list` or :obj:`NoneType`, defaults to None
            Variables to check if given input contains them.
            This is set to None by default.

        Returns
        -------
        input_dict : :obj:`SampleDict`
            Variables checked in this method.

        Raises
        ------
        ValueError
            Raises `ValueError` if the type of input is neither :obj:`torch.Tensor`, :obj:`list`, nor :obj:`dict.

        """
        if not input_:
            return {}

        if var is None:
            var = self.input_var

        if torch.is_tensor(input_):
            input_dict = {var[0]: input_}

        elif isinstance(input_, list):
            # TODO: we need to check if all the elements contained in this list are torch.Tensor.
            input_dict = dict(zip(var, input_))

        elif isinstance(input_, dict):
            if not (set(list(input_.keys())) >= set(var)):
                raise ValueError("Input keys are not valid.")
            input_dict = input_.copy()

        else:
            raise ValueError(f"The type of input is not valid, got {type(input_)}.")

        return input_dict

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        """Sample variables of this distribution.
        If :attr:`cond_var` is not empty, you should set inputs as :obj:`dict`.

        Parameters
        ----------
        x_dict : :obj:`torch.Tensor`, :obj:`list`, or :obj:`dict`, or :obj:`SampleDict`, defaults to {}
            Input variables.
        sample_shape : :obj:`list` or :obj:`NoneType`, defaults to torch.Size()
            Shape of generating samples.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        return_all : :obj:`bool`, defaults to True
            Choose whether the output contains input variables.
        reparam : :obj:`bool`, defaults to False.
            Choose whether we sample variables with re-parameterized trick.

        Returns
        -------
        output : :obj:`SampleDict`
            Samples of this distribution.

        Examples
        --------
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...            features_shape=[10, 2])
        >>> print(p)
        Distribution:
          p(x)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([10, 2])
            (loc): torch.Size([1, 10, 2])
            (scale): torch.Size([1, 10, 2])
          )
        >>> p.sample()["x"].shape  # (batch_n=1, features_shape)
        torch.Size([1, 10, 2])
        >>> p.sample(batch_n=20)["x"].shape  # (batch_n, features_shape)
        torch.Size([20, 10, 2])
        >>> p.sample(batch_n=20, sample_shape=[40, 30])["x"].shape  # (sample_shape, batch_n, features_shape)
        torch.Size([40, 30, 20, 10, 2])

        >>> # Conditional distribution
        >>> p = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...            features_shape=[10])
        >>> print(p)
        Distribution:
          p(x|y)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([10])
            (scale): torch.Size([1, 10])
          )
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> sample_a = torch.randn(1, 10) # Psuedo data
        >>> sample = p.sample({"y": sample_y})
        >>> print(sample) # input_var + var  # doctest: +SKIP
        {'y': tensor([[-0.5182,  0.3484,  0.9042,  0.1914,  0.6905,
                       -1.0859, -0.4433, -0.0255, 0.8198,  0.4571]]),
         'x': tensor([[-0.7205, -1.3996,  0.5528, -0.3059,  0.5384,
                       -1.4976, -0.1480,  0.0841,0.3321,  0.5561]])}
        >>> sample = p.sample({"y": sample_y, "a": sample_a}) # Redundant input ("a")
        >>> print(sample) # input_var + var + "a" (redundant input)  # doctest: +SKIP
        {'y': tensor([[ 1.3582, -1.1151, -0.8111,  1.0630,  1.1633,
                        0.3855,  2.6324, -0.9357, -0.8649, -0.6015]]),
         'a': tensor([[-0.1874,  1.7958, -1.4084, -2.5646,  1.0868,
                       -0.7523, -0.0852, -2.4222, -0.3914, -0.9755]]),
         'x': tensor([[-0.3272, -0.5222, -1.3659,  1.8386,  2.3204,
                        0.3686,  0.6311, -1.1208, 0.3656, -0.6683]])}

        """
        raise NotImplementedError()

    def sample_mean(self, x_dict=None):
        """Return the mean of the distribution.

        Parameters
        ----------
        x_dict : :obj:`dict`, or :obj:`SampleDict`, defaults to {}
            Parameters of this distribution.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> mean = p1.sample_mean()
        >>> print(mean)
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> mean = p2.sample_mean({"y": sample_y})
        >>> print(mean) # doctest: +SKIP
        tensor([[-0.2189, -1.0310, -0.1917, -0.3085,  1.5190, -0.9037,  1.2559,  0.1410,
                  1.2810, -0.6681]])

        """
        raise NotImplementedError()

    def sample_variance(self, x_dict=None):
        """Return the variance of the distribution.

        Parameters
        ----------
        x_dict : :obj:`dict`, :obj:`SampleDict`, defaults to {}
            Parameters of this distribution.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> var = p1.sample_variance()
        >>> print(var)
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> var = p2.sample_variance({"y": sample_y})
        >>> print(var) # doctest: +SKIP
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        """
        raise NotImplementedError()

    def get_log_prob(self, x_dict):
        """Giving variables, this method returns values of log-pdf.

        Parameters
        ----------
        x_dict : :obj:`dict`, or :obj:`SampleDict`
            Input variables.
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some dimensions which are specified by `feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output.

        Returns
        -------
        log_prob : torch.Tensor
            Values of log-probability density/mass function.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p1.log_prob({"x": sample_x})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-16.1153])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p2.log_prob({"x": sample_x, "y": sample_y})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-21.5251])

        """
        raise NotImplementedError()

    def get_entropy(self, x_dict=None):
        """Giving variables, this method returns values of entropy.

        Parameters
        ----------
        x_dict : :obj:`dict`, or :obj:`SampleDict`, defaults to {}
            Input variables.
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some dimensions which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output.

        Returns
        -------
        entropy : torch.Tensor
            Values of entropy.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> entropy = p1.get_entropy()
        >>> print(entropy)
        tensor([14.1894])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> entropy = p2.get_entropy({"y": sample_y})
        >>> print(entropy)
        tensor([14.1894])

        """
        raise NotImplementedError()

    def log_prob(self):
        """Return an instance of :class:`pixyz.losses.LogProb`.

        Parameters
        ----------
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some axes (dimensions) which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set axes to sum across the output.

        Returns
        -------
        pixyz.losses.LogProb
            An instance of :class:`pixyz.losses.LogProb`

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p1.log_prob().eval({"x": sample_x})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-16.1153])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p2.log_prob().eval({"x": sample_x, "y": sample_y})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-21.5251])

        """
        return LogProb(self)

    def prob(self):
        """Return an instance of :class:`pixyz.losses.LogProb`.

        Parameters
        ----------
        sum_features : :obj:`bool`, defaults to True
            Choose whether the output is summed across some axes (dimensions)
            which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output. (Note: this parameter is not used for now.)

        Returns
        -------
        pixyz.losses.Prob
            An instance of :class:`pixyz.losses.Prob`

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> prob = p1.prob().eval({"x": sample_x})
        >>> print(prob) # doctest: +SKIP
        tensor([4.0933e-07])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> prob = p2.prob().eval({"x": sample_x, "y": sample_y})
        >>> print(prob) # doctest: +SKIP
        tensor([2.9628e-09])

        """
        return Prob(self)

    def forward(self, *args, **kwargs):
        """When this class is inherited by DNNs, this method should be overrided."""

        raise NotImplementedError()

    def replace_var(self, **replace_dict):
        """Return an instance of :class:`pixyz.distributions.ReplaceVarDistribution`.

        Parameters
        ----------
        replace_dict : dict
            Dictionary.

        Returns
        -------
        pixyz.distributions.ReplaceVarDistribution
            An instance of :class:`pixyz.distributions.ReplaceVarDistribution`

        """

        return ReplaceVarDistribution(self, replace_dict)

    def marginalize_var(self, marginalize_list):
        """Return an instance of :class:`pixyz.distributions.MarginalizeVarDistribution`.

        Parameters
        ----------
        marginalize_list : :obj:`list` or other
            Variables to marginalize.

        Returns
        -------
        pixyz.distributions.MarginalizeVarDistribution
            An instance of :class:`pixyz.distributions.MarginalizeVarDistribution`

        """

        marginalize_list = tolist(marginalize_list)
        return MarginalizeVarDistribution(self, marginalize_list)

    def __mul__(self, other):
        return MultiplyDistribution(self, other)

    def __str__(self):
        # Distribution
        text = "Distribution:\n  {}\n".format(self.prob_joint_factorized_and_text)

        # Network architecture (`repr`)
        network_text = self.__repr__()
        network_text = re.sub('^', ' ' * 2, str(network_text), flags=re.MULTILINE)
        text += "Network architecture:\n{}".format(network_text)
        return text

    def extra_repr(self):
        # parameters
        parameters_text = f'name={self.name}, distribution_name={self.distribution_name},\n' \
                          f'var={self.var}, cond_var={self.cond_var}, input_var={self.input_var}'

        if len(self._buffers) != 0:
            # add buffers to repr
            buffers = ["({}): {}".format(key, value.shape) for key, value in self._buffers.items()]
            return parameters_text + "\n" + "\n".join(buffers)

        return parameters_text


class DistributionBase(Distribution):
    """Distribution class with PyTorch. In Pixyz, all distributions are required to inherit this class."""

    def __init__(self, cond_var=(), var=("x",), name="p", features_shape=None, **params_dict):
        super().__init__(cond_var=cond_var, var=var, name=name)
        if len(var) != 1:
            raise ValueError("multiple var distribution is not supported.")

        self._set_buffers(**params_dict)
        self._dist = None
        # if len(iid_dims) == 0:
        #     iid_dims = range(len(iid_shape))
        # elif len(iid_dims) != len(iid_shape):
        #     raise ValueError("the length of iid_dims and iid_shape must be the same.")
        # self.iid_info = (iid_dims, iid_shape)
        self._features_shape = features_shape
        self._is_iid_dist = (features_shape is None)

    @property
    def features_shape(self):
        if not self._features_shape:
            raise Exception("features_shape is invalid until the distribution is sampled or its log_prob is evaluated.")
        return self._features_shape

    def _set_buffers(self, **params_dict):
        """Format constant parameters of this distribution as buffers.

        Parameters
        ----------
        params_dict : dict of (str or torch.Tensor)
            Constant parameters of this distribution set at initialization.
            If the values of these dictionaries contain parameters which are named as strings, which means that
            these parameters are set as `variables`, the correspondences between these values and the true name of
            these parameters are stored as :obj:`dict` (:attr:`replace_params_dict`).

        """

        self.replace_params_dict = {}

        for key, value in params_dict.items():
            if isinstance(value, str):
                if value not in self._cond_var:
                    raise ValueError(f"a given parameter {value} is not in cond_var of the distribution.")
                self.replace_params_dict[value] = key
            elif torch.is_tensor(value):
                self.register_buffer(key, value)
            else:
                raise ValueError(f"only Tensor or str parameters are supported. ({key}:{type(value)} is given.)")

    @property
    def params_keys(self):
        """list: Return the list of parameter names for this distribution."""
        raise NotImplementedError()

    @property
    def distribution_torch_class(self):
        """Return the class of PyTorch distribution."""
        raise NotImplementedError()

    @property
    def dist(self):
        """Return the instance of PyTorch distribution."""
        return self._dist

    def set_dist(self, x_dict=None, relaxing=False):
        """Set :attr:`dist` as PyTorch distributions given parameters.

        This requires that :attr:`params_keys` and :attr:`distribution_torch_class` are set.

        Parameters
        ----------
        x_dict : :obj:`dict`, or :obj:`SampleDict`, defaults to {}.
            Parameters of this distribution.
        relaxing : :obj:`bool`, defaults to False.
            Choose whether to use relaxed_* in PyTorch distribution.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        params = self.get_params(x_dict)
        # if set(self.params_keys) != set(params.keys()):
        #     raise ValueError(f"params keys don't match. expected: {self.params_keys}, actual: {params.keys()}")
        input_sample_shape = params.sample_shape

        # self._dist_shape = SampleDict.max_shape_(params)
        self._dist = self.distribution_torch_class(**params)
        # iid_dims, iid_shape = self.iid_info
        # TODO: iidの問題が未解決（paramsのexpandでdistribution expandでのshapeの問題を解決したが，MultidimentionalNormalDistributionでのiid指定ができない＋今までの面倒なshapeソートの名残が残っている）
        if self._is_iid_dist:
            if self._dist.batch_shape != input_sample_shape:
                raise ValueError("torch distribution got wrong parameter which has too many features_shape.")
            self._dist = self.distribution_torch_class(**params)
            self._dist.expand(input_sample_shape + self.features_shape)
        else:
            # -squeeze可能な次元に吸収されてしまう問題があった（今はshape全体を指定しているので無い）
            # expand _dist for iid distribution and ancestral sampling
            self._dist.expand(input_sample_shape + self._dist.batch_shape[len(input_sample_shape):])

            # SampleDist.features_dims(params, self.var[0]) can not be used because params don't have self.var[0]
            features_dims = (SampleDict.sample_dims_(params)[-1], None)
            # TODO: shapeの型をtorch.Size()に揃えるべきか？
            self._features_shape = self._dist.batch_shape[slice(*features_dims)] + self._dist.event_shape
            # TODO: features_shapeをevent_shapeにリネームしたくなってきた(pytorchと統一)

            # -本当にbatch_nオプションを削除してよかったのか確認する

    def _get_sample(self, reparam=False, sample_shape=torch.Size()):
        """Get a sample_shape shaped sample from :attr:`dist`.

        Parameters
        ----------
        reparam : :obj:`bool`, defaults to True.
            Choose where to sample using re-parameterization trick.

        sample_shape : :obj:`tuple` or :obj:`torch.Size`, defaults to torch.Size().
            Set the shape of a generated sample.

        Returns
        -------
        output_dict : :obj:`dict`
            Generated sample.

        """
        if reparam:
            try:
                samples = self.dist.rsample(sample_shape=sample_shape)
            except NotImplementedError():
                raise ValueError("You cannot use the re-parameterization trick for this distribution.")
        else:
            samples = self.dist.sample(sample_shape=sample_shape)

        return samples

    def sum_over_iid_dims(self, torch_loss, given_dict):
        x_target = given_dict[self.var[0]]
        new_ndim = x_target.ndim - len(self._dist.batch_shape) - len(self._dist.event_shape)
        # sum over iid dims
        torch_loss = torch_loss.sum(dim=range(new_ndim, new_ndim + len(self.iid_info[0])))
        return torch_loss

    # TODO: 本当はここにSampleDictやproduct_infoに関する知識を含めないことで拡張しやすくしたいが，難しい
    def get_log_prob(self, x_dict):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.var + self._cond_var)
        _x_dict = x_dict.from_variables(self._cond_var)
        self.set_dist(_x_dict)

        x_target = x_dict[self.var[0]]
        # TODO: -iid_shapeについてソートしていない
        log_prob = self.dist.log_prob(x_target)
        # log_prob = self.sum_over_iid_dims(log_prob, x_dict)

        return log_prob

    def get_params(self, params_dict=None):
        """This method aims to get parameters of this distributions from constant parameters set in initialization
        and outputs of DNNs.

        Parameters
        ----------
        params_dict : :obj:`dict`, or :obj:`SampleDict`, defaults to {}
            Input parameters.

        Returns
        -------
        output_dict : :obj:`SampleDict`
            Output parameters.

        Examples
        --------
        >>> from pixyz.distributions import Normal
        >>> dist_1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...                 features_shape=[1])
        >>> print(dist_1)
        Distribution:
          p(x)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
        >>> dist_1.get_params()
        {'loc': tensor([[0.]]) --(shape=[('batch', [1]), ('feature', [1])]), 'scale': tensor([[1.]]) --(shape=[('batch', [1]), ('feature', [1])])}

        >>> dist_2 = Normal(loc=torch.tensor(0.), scale="z", cond_var=["z"], var=["x"])
        >>> print(dist_2)
        Distribution:
          p(x|z)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
            (loc): torch.Size([1])
          )
        >>> dist_2.get_params({"z": torch.tensor(1.)})
        {'scale': tensor(1.) --(shape=[]), 'loc': tensor([0.]) --(shape=[('feature', [1])])}

        """
        params_dict, vars_dict = SampleDict.split_(params_dict, self.replace_params_dict.keys())
        params_dict = SampleDict.replaced_dict_(params_dict, self.replace_params_dict)

        output_dict = self.forward(**vars_dict)
        params_dict.update(output_dict)

        # append constant parameters to output_dict
        constant_params_dict = SampleDict({**self.named_buffers()}).from_variables(self.params_keys)
        params_dict.update(constant_params_dict)

        # unsqueeze params for iid dims
        if self._is_iid_dist:
            for key in params_dict.keys():
                params_dict[key] = params_dict[key][
                    (slice(None),)*params_dict[key].ndim + (None,)*len(self.features_shape)]

        return params_dict

    def get_entropy(self, x_dict=None):
        # TODO: いくつかの派生先ではkwargsが使われていたのでチェックする
        x_dict = SampleDict.from_arg(x_dict, required_keys=self._cond_var)
        _x_dict = x_dict.from_variables(self._cond_var)
        self.set_dist(_x_dict, relaxing=False)

        entropy = self.dist.entropy()
        # sum over iid dims
        entropy = entropy.sum(dim=range(len(self.iid_info[0])))
        # TODO: sum_featuresオプションをなくして（旧来実装の互換が）大丈夫だったか？

        return entropy

    def sort_iid_dims(self, torch_sample, sample_ndim=0):
        offset = sample_ndim
        for dim, iid_dim in enumerate(self.iid_info[0]):
            torch_sample = torch_sample.transpose(offset + dim, offset + iid_dim)
        return torch_sample

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        # global_sample: x_dictに含まれるinput_sample_shapeについて，ブロードキャストではなくサンプルするかどうか
        # check whether the input is valid or convert it to valid dictionary.
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.input_var)
        sample_shape = torch.Size(sample_shape)

        # conditioned
        input_dict = x_dict.from_variables(self.input_var)
        input_sample_shape = input_dict.sample_shape
        # 辞書内でブロードキャストしきらない場合があり得るので，shapeをここで統一する必要があるかも
        # input_sample_shapeについて，sampleする，unsqueezeだけでなくexpandする

        # batch_n（名前はn_batchのほうが良さそう）をsample_shapeに統一しているが，sample_shapeと微妙に異なる点がある
        # それはinput_dictの段階でunsqueeze(+expand)されているのが基本だということだ，set_distから分かる
        # 重大な問題として新規shapeがunsqueezeなのかexpandなのか区別する必要がある．
        # ちなみにdist.expandは足りない分を自動でunsqueezeする
        # batch_shapeは廃止できない，2つ目の事前分布がdictからパラメータを受け取るとbatch_dimがある
        # SampleDictの仕様を変更し，unsqueezeせずにgivenするようにすると，sample_shapeのみで良くなるが
        # 自分で(4,1,1,1)のようにユーザーが型を合わせる必要が出てくる
        self.set_dist(input_dict)
        sample = self._get_sample(reparam=reparam, sample_shape=sample_shape)
        # output shape is (sample_shape, iid_shape, input_sample_shape, other_features_shape).

        # iid_shape is restricted to be located at head of shape in pytorch distribution.
        sample = self.sort_iid_dims(sample, sample_ndim=len(sample_shape))
        # output shape is (sample_shape, input_sample_shape, features_shape(iid_shape & other_features_shape)).

        x_dict.update(SampleDict({self.var[0]: sample}, sample_shape=sample_shape + input_sample_shape))

        return x_dict if return_all else x_dict.from_variables(self.var)

    def sample_mean(self, x_dict=None):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.input_var)
        input_sample_dims = x_dict.sample_dims
        self.set_dist(x_dict)
        return self.sort_iid_dims(self.dist.mean, sample_ndim=input_sample_dims[1])

    def sample_variance(self, x_dict=None):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.input_var)
        input_sample_dims = x_dict.sample_dims
        self.set_dist(x_dict)
        return self.sort_iid_dims(self.dist.variance, sample_ndim=input_sample_dims[1])

    def forward(self, **params):
        return params


class MultiplyDistribution(Distribution):
    """Multiply by given distributions, e.g, :math:`p(x,y|z) = p(x|z,y)p(y|z)`.
    In this class, it is checked if two distributions can be multiplied.

    p(x|z)p(z|y) -> Valid

    p(x|z)p(y|z) -> Valid

    p(x|z)p(y|a) -> Valid

    p(x|z)p(z|x) -> Invalid (recursive)

    p(x|z)p(x|y) -> Invalid (conflict)

    Examples
    --------
    >>> a = DistributionBase(var=["x"], cond_var=["z"])
    >>> b = DistributionBase(var=["z"], cond_var=["y"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,z|y) = p(x|z)p(z|y)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['z'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([])
      )
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"], cond_var=["z"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"], cond_var=["a"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z,a) = p(x|z)p(y|a)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['a'], input_var=['a'], features_shape=torch.Size([])
      )
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )

    """

    def __init__(self, a, b):
        """
        Parameters
        ----------
        a : pixyz.Distribution
            Distribution.

        b : pixyz.Distribution
            Distribution.

        """
        if not (isinstance(a, Distribution) and isinstance(b, Distribution)):
            raise ValueError("Given inputs should be `pixyz.Distribution`, got {} and {}.".format(type(a), type(b)))

        # Check parent-child relationship between two distributions.
        # If inherited variables (`_inh_var`) are exist (e.g. c in p(e|c)p(c|a,b)),
        # then p(e|c) is a child and p(c|a,b) is a parent, otherwise it is opposite.
        _vars_a_b = a.cond_var + b.var
        _vars_b_a = b.cond_var + a.var
        _inh_var_a_b = [var for var in set(_vars_a_b) if _vars_a_b.count(var) > 1]
        _inh_var_b_a = [var for var in set(_vars_b_a) if _vars_b_a.count(var) > 1]

        if len(_inh_var_a_b) > 0:
            _child = a
            _parent = b
            _inh_var = _inh_var_a_b

        elif len(_inh_var_b_a) > 0:
            _child = b
            _parent = a
            _inh_var = _inh_var_b_a

        else:
            _child = a
            _parent = b
            _inh_var = []

        # Check if variables of two distributions are "recursive" (e.g. p(x|z)p(z|x)).
        _check_recursive_vars = _child.var + _parent.cond_var
        if len(_check_recursive_vars) != len(set(_check_recursive_vars)):
            raise ValueError("Variables of two distributions, {} and {}, are recursive.".format(_child.prob_text,
                                                                                                _parent.prob_text))

        # Set variables.
        _var = _child.var + _parent.var
        if len(_var) != len(set(_var)):  # e.g. p(x|z)p(x|y)
            raise ValueError("Variables of two distributions, {} and {}, are conflicted.".format(_child.prob_text,
                                                                                                 _parent.prob_text))

        # Set conditional variables.
        _cond_var = _child.cond_var + _parent.cond_var
        _cond_var = sorted(set(_cond_var), key=_cond_var.index)

        # Delete inh_var in conditional variables.
        _cond_var = [var for var in _cond_var if var not in _inh_var]

        super().__init__(cond_var=_cond_var, var=_var)

        self._parent = _parent
        self._child = _child
        self._inh_var = _inh_var

        # Set input_var (it might be different from cond_var if either a and b contain data distributions.)
        _input_var = [var for var in self._child.input_var if var not in _inh_var]
        _input_var += self._parent.input_var
        self._input_var = sorted(set(_input_var), key=_input_var.index)

    @property
    def input_var(self):
        return self._input_var

    @property
    def prob_factorized_text(self):
        return self._child.prob_factorized_text + self._parent.prob_factorized_text

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.input_var)
        # sample from the parent distribution
        parents_x_dict = x_dict
        child_x_dict = self._parent.sample(x_dict=parents_x_dict, sample_shape=sample_shape,
                                           return_all=True, reparam=reparam)
        if not isinstance(child_x_dict, SampleDict):
            raise ValueError("result of sample must be a instance of SampleDict.")

        # # もしかしてsplit必要ない？squeezeしてないし・・・
        # child_x_dict, output_dict = output_dict.split(self._inh_var)
        # # if parent and child are independent, sample_shape should be set separately.
        # child_sample_shape = sample_shape if self._inh_var else torch.Size()
        # sample from the child distribution
        # output_dict2 = self._child.sample(x_dict=child_x_dict, sample_shape=child_sample_shape,
        #                                   return_all=True, reparam=reparam)
        output_dict = self._child.sample(x_dict=child_x_dict, return_all=True, reparam=reparam)
        if not isinstance(output_dict, SampleDict):
            raise ValueError("result of sample must be a instance of SampleDict.")
        # TODO: 連鎖部分では必ず型チェックする，数が多くなるようなら検討する-> 入力制限の法が良いかも
        # output_dict.update(output_dict2)

        return output_dict if return_all else output_dict.from_variables(self._var)

    def get_log_prob(self, x_dict):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.var + self.cond_var)
        parent_log_prob = self._parent.get_log_prob(x_dict)
        child_log_prob = self._child.get_log_prob(x_dict)

        if parent_log_prob.size() == child_log_prob.size():
            return parent_log_prob + child_log_prob

        raise ValueError("Two PDFs, {} and {}, have different sizes,"
                         " so you must set sum_dim=True.".format(self._parent.prob_text, self._child.prob_text))

    def __repr__(self):
        return self._parent.__repr__() + "\n" + self._child.__repr__()


class ReplaceVarDistribution(Distribution):
    """Replace names of variables in Distribution.

    Examples
    --------
    >>> p = DistributionBase(var=["x"], cond_var=["z"])
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> replace_dict = {'x': 'y'}
    >>> p_repl = ReplaceVarDistribution(p, replace_dict)
    >>> print(p_repl)
    Distribution:
      p(y|z)
    Network architecture:
      ReplaceVarDistribution(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (p): DistributionBase(
          name=p, distribution_name=,
          var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        )
      )

    """

    def __init__(self, p, replace_dict):
        """
        Parameters
        ----------
        p : :class:`pixyz.distributions.Distribution` (not :class:`pixyz.distributions.MultiplyDistribution`)
            Distribution.

        replace_dict : dict
            Dictionary.

        """
        if not isinstance(p, Distribution):
            raise ValueError("Given input should be `pixyz.Distribution`, got {}.".format(type(p)))

        if isinstance(p, MultiplyDistribution):
            raise ValueError("`pixyz.MultiplyDistribution` is not supported for now.")

        if isinstance(p, MarginalizeVarDistribution):
            raise ValueError("`pixyz.MarginalizeVarDistribution` is not supported for now.")

        _cond_var = deepcopy(p.cond_var)
        _var = deepcopy(p.var)
        all_vars = _cond_var + _var

        if not (set(replace_dict.keys()) <= set(all_vars)):
            raise ValueError()

        _replace_inv_cond_var_dict = {replace_dict[var]: var for var in _cond_var if var in replace_dict.keys()}
        _replace_inv_dict = {value: key for key, value in replace_dict.items()}

        self._replace_inv_cond_var_dict = _replace_inv_cond_var_dict
        self._replace_inv_dict = _replace_inv_dict
        self._replace_dict = replace_dict

        _cond_var = [replace_dict[var] if var in replace_dict.keys() else var for var in _cond_var]
        _var = [replace_dict[var] if var in replace_dict.keys() else var for var in _var]
        super().__init__(cond_var=_cond_var, var=_var, name=p.name)

        self.p = p
        _input_var = [replace_dict[var] if var in replace_dict.keys() else var for var in p.input_var]
        self._input_var = _input_var

    def forward(self, *args, **kwargs):
        return self.p.forward(*args, **kwargs)

    def get_params(self, params_dict=None):
        params_dict = SampleDict.replaced_dict_(params_dict, self._replace_inv_cond_var_dict)
        return self.p.get_params(params_dict)

    def set_dist_and_shape(self, x_dict=None, relaxing=False):
        x_dict = SampleDict.replaced_dict_(x_dict, self._replace_inv_cond_var_dict)
        return self.p.set_dist(x_dict=x_dict, relaxing=relaxing)

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.input_var)
        input_dict = x_dict.from_variables(self.cond_var)
        replaced_input_dict = input_dict.replaced_dict(self._replace_inv_cond_var_dict)

        output_dict = self.p.sample(replaced_input_dict, sample_shape=sample_shape,
                                    return_all=False, reparam=reparam)
        output_dict = output_dict.replaced_dict(self._replace_dict)

        x_dict.update(output_dict)
        return x_dict

    def get_log_prob(self, x_dict):
        x_dict = SampleDict.from_arg(x_dict, require_keys=self.var + self.cond_var)
        input_dict = x_dict.from_variables(self.cond_var + self.var)
        input_dict = input_dict.dict_with_replaced_keys(self._replace_inv_dict)
        return self.p.get_log_prob(input_dict)

    def sample_mean(self, x_dict=None):
        x_dict = SampleDict.from_arg(x_dict, require_keys=self.input_var)
        input_dict = x_dict.from_variables(self.cond_var)
        input_dict = input_dict.replaced_dict(self._replace_inv_cond_var_dict)
        return self.p.sample_mean(input_dict)

    def sample_variance(self, x_dict=None):
        x_dict = SampleDict.from_arg(x_dict, require_keys=self.input_var)
        input_dict = x_dict.from_variables(self.cond_var)
        input_dict = input_dict.replaced_dict(self._replace_inv_cond_var_dict)
        return self.p.sample_variance(input_dict)

    @property
    def input_var(self):
        return self._input_var

    @property
    def distribution_name(self):
        return self.p.distribution_name

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self.p.__getattribute__(item)


class MarginalizeVarDistribution(Distribution):
    r"""Marginalize variables in Distribution.

    .. math::
        p(x) = \int p(x,z) dz

    Examples
    --------
    >>> a = DistributionBase(var=["x"], cond_var=["z"])
    >>> b = DistributionBase(var=["y"], cond_var=["z"])
    >>> p_multi = a * b
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> p_marg = MarginalizeVarDistribution(p_multi, ["y"])
    >>> print(p_marg)
    Distribution:
      p(x|z) = \int p(x|z)p(y|z)dy
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )

    """

    def __init__(self, p, marginalize_list):
        """
        Parameters
        ----------
        p : :class:`pixyz.distributions.Distribution` (not :class:`pixyz.distributions.DistributionBase`)
            Distribution.

        marginalize_list : list
            Variables to marginalize.

        """
        marginalize_list = tolist(marginalize_list)

        if not isinstance(p, Distribution):
            raise ValueError("Given input must be `pixyz.distributions.Distribution`, got {}.".format(type(p)))

        if isinstance(p, DistributionBase):
            raise ValueError("`pixyz.distributions.DistributionBase` cannot be marginalized its variables.")

        _var = deepcopy(p.var)
        _cond_var = deepcopy(p.cond_var)

        if not((set(marginalize_list)) < set(_var)):
            raise ValueError()

        if not((set(marginalize_list)).isdisjoint(set(_cond_var))):
            raise ValueError()

        if len(marginalize_list) == 0:
            raise ValueError("Length of `marginalize_list` must be at least 1, got 0.")

        _var = [var for var in _var if var not in marginalize_list]

        super().__init__(cond_var=_cond_var, var=_var, name=p.name)
        self.p = p
        self._marginalize_list = marginalize_list

    def forward(self, *args, **kwargs):
        return self.p.forward(*args, **kwargs)

    def get_params(self, params_dict=None):
        return self.p.get_params(params_dict)

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        output_dict = self.p.sample(x_dict=x_dict, sample_shape=sample_shape, return_all=return_all,
                                    reparam=reparam)
        _, output_dict = output_dict.split(self._marginalize_list)

        return output_dict

    def sample_mean(self, x_dict=None):
        return self.p.sample_mean(x_dict)

    def sample_variance(self, x_dict=None):
        return self.p.sample_variance(x_dict)

    @property
    def input_var(self):
        return self.p.input_var

    @property
    def distribution_name(self):
        return self.p.distribution_name

    @property
    def prob_factorized_text(self):
        integral_symbol = len(self._marginalize_list) * "\\int "
        integral_variables = ["d" + str(var) for var in self._marginalize_list]
        integral_variables = "".join(integral_variables)

        return "{}{}{}".format(integral_symbol, self.p.prob_factorized_text, integral_variables)

    def __repr__(self):
        return self.p.__repr__()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self.p.__getattribute__(item)
