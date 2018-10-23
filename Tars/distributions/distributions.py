from __future__ import print_function
import torch
from torch import nn

from ..utils import get_dict_values


class Distribution(nn.Module):
    """
    Distribution class. In Tars, all distributions are required to inherit this class.

    Attributes
    ----------
    var : list
        Variables of this distribution.

    cond_var : list
        Conditional variables of this distribution.
        In case that cond_var is not empty, we must set the corresponding inputs in order to
        sample variables or estimate the log likelihood.

    dim : int
        Number of dimensions of this distribution.
        This might be ignored depending on the shape which is set in the sample method and on its parent distribution.
        Moreover, this is not consider when this class is inherited by DNNs.
        This is set to 1 by default.

    name : str
        Name of this distribution.
        This name is displayed in prob_text and prob_factorized_text.
        This is set to "p" by default.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=1,
                 **kwargs):
        super().__init__()
        self._cond_var = cond_var
        self._var = var
        self.dim = dim
        self._name = name

        self._prob_text = None
        self._prob_factorized_text = None

        # these members are intended to be overrided.
        # self.dist = None
        # self.distribution_name = None
        # self.params_keys = None

        self._initialize_constant_params(**kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if type(name) is str:
            self._name = name
            self._update_prob_text()
            return

        raise ValueError("Name of the distribution class must be set as a string type.")

    @property
    def var(self):
        return self._var

    @property
    def cond_var(self):
        return self._cond_var

    @property
    def prob_text(self):
        _var_text = [','.join(self._var)]
        if len(self._cond_var) != 0:
            _var_text += [','.join(self._cond_var)]

        _prob_text = "{}({})".format(
            self._name,
            "|".join(_var_text)
        )

        return _prob_text

    @property
    def prob_factorized_text(self):
        return self.prob_text

    def _check_input(self, x, var=None):
        """
        Check the type of a given input.
        If this type is a dictionary, we check whether this key and `var` are same.
        In case that this is list or tensor, we return a output formatted in a dictionary.

        Parameters
        ----------
        x : torch.Tensor, list, or dict
            Input variables

        var : list or None
            Variables to check if `x` has them.
            This is set to None by default.

        Returns
        -------
        checked_x : dict
            Variables which are checked in this method.

        Raises
        ------
        ValueError
            Raises ValueError if the type of `x` is neither tensor, list, nor dictionary.
        """

        if var is None:
            var = self._cond_var

        if type(x) is torch.Tensor:
            checked_x = {var[0]: x}

        elif type(x) is list:
            checked_x = dict(zip(var, x))

        elif type(x) is dict:
            if not set(list(x.keys())) == set(var):
                raise ValueError("Input's keys are not valid.")
            checked_x = x

        else:
            raise ValueError("The type of input is not valid, got %s."
                             % type(x))

        return checked_x

    def _initialize_constant_params(self, **params):
        """
        Format constant parameters set at initialization of this distribution.

        Parameters
        ----------
        params : dict
            Constant parameters of this distribution set at initialization.
            If the values of these dictionaries contain parameters which are named as strings, which means that
            these parameters are set as "variables", the correspondences between these values and the true name of
            these parameters are stored as a dictionary format (`map_dict`).
        """

        self.constant_params = {}
        self.map_dict = {}

        for keys in self.params_keys:
            if keys in params.keys():
                if type(params[keys]) is str:
                    self.map_dict[params[keys]] = keys
                else:
                    self.constant_params[keys] = params[keys]

        # Set a distribution if all parameters are constant and
        # set at initialization.
        if len(self.constant_params) == len(self.params_keys):
            self._set_distribution()

    def _set_distribution(self, x={}, **kwargs):
        params = self.get_params(x, **kwargs)
        self.dist = self.DistributionTorch(**params)

    def _get_sample(self, reparam=True,
                    sample_shape=torch.Size()):
        """
        Parameters
        ----------
        reparam : bool

        sample_shape : tuple

        Returns
        -------
        samples_dict : dict

        """

        if reparam:
            try:
                _samples = self.dist.rsample(sample_shape=sample_shape)
            except NotImplementedError:
                print("We can not use the reparameterization trick"
                      "for this distribution.")
        else:
            _samples = self.dist.sample(sample_shape=sample_shape)
        samples_dict = {self._var[0]: _samples}

        return samples_dict

    def _get_log_like(self, x):
        """
        Parameters
        ----------
        x : dict

        Returns
        -------
        log_like : torch.Tensor

        """

        x_targets = get_dict_values(x, self._var)
        log_like = self.dist.log_prob(*x_targets)

        return log_like

    def _map_variables_to_params(self, **variables):
        """
        Replace variables in keys of a input dictionary to parameters of this distribution according to
        these correspondences which is formatted in a dictionary and set in `_initialize_constant_params`.

        Parameters
        ----------
        variables : dict

        Returns
        -------
        mapped_params : dict

        variables : dict

        Examples
        --------
        >> distribution.map_dict
        >> > {"a": "loc"}
        >> x = {"a": 0}
        >> distribution._map_variables_to_params(x)
        >> > {"loc": 0}, {}
        """

        mapped_params = {self.map_dict[key]: value for key, value in variables.items()
                         if key in list(self.map_dict.keys())}

        variables = {key: value for key, value in variables.items()
                     if key not in list(self.map_dict.keys())}

        return mapped_params, variables

    def get_params(self, params):
        """
        This method aims to get parameters of this distributions from constant parameters set in
        initialization and outputs of DNNs.

        Parameters
        ----------
        params : dict

        Returns
        -------
        output : dict

        Examples
        --------
        >> print(dist_1.prob_text, dist_1.distribution_name)
        >> > p(x) Normal
        >> dist_1.get_params()
        >> > {"loc": 0, "scale": 1}
        >> print(dist_2.prob_text, dist_2.distribution_name)
        >> > p(x|z) Normal
        >> dist_1.get_params({"z": 1})
        >> > {"loc": 0, "scale": 1}
        """

        params, variables = self._map_variables_to_params(**params)
        output = self.forward(**variables)

        # append constant_params to dict
        output.update(params)
        output.update(self.constant_params)

        return output

    def sample(self, x=None, shape=None, batch_size=1, return_all=True,
               reparam=True, **kwargs):
        """
        Sample variables of this distribution.
        If `cond_var` is not empty, we should set inputs as a dictionary format.

        Parameters
        ----------
        x : torch.Tensor, list, or dict
            Input variables.

        shape : tuple
            Shape of samples.
            If set, `batch_size` and `dim` are ignored.

        batch_size : int
            Batch size of samples. This is set to 1 by default.

        return_all : bool
            Choose whether the output contains input variables.

        reparam : bool
            Choose whether we sample variables with reparameterized trick.

        kwargs : dict

        Returns
        -------
        output : dict
            Samples of this distribution.
        """

        if x is None:  # unconditioned
            if len(self._cond_var) != 0:
                raise ValueError("You should set inputs or parameters")

            if shape:
                sample_shape = shape
            else:
                sample_shape = (batch_size, self.dim)

            output = self._get_sample(reparam=reparam,
                                      sample_shape=sample_shape)

        else:  # conditioned
            x = self._check_input(x)
            self._set_distribution(x, **kwargs)
            output = self._get_sample(reparam=reparam)

            if return_all:
                output.update(x)

        return output

    def log_likelihood(self, x):
        """
        Estimate the log likelihood of this distribution from inputs formatted by a dictionary.

        Parameters
        ----------
        x : dict


        Returns
        -------
        log_like : torch.Tensor

        """

        if not set(list(x.keys())) >= set(self._cond_var + self._var):
            raise ValueError("Input's keys are not valid.")

        if len(self._cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self._cond_var, True)
            self._set_distribution(_x)

        log_like = self._get_log_like(x)
        log_like = mean_sum_samples(log_like)
        return log_like

    def forward(self, **params):
        """
        When this class is inherited by DNNs, it is also intended that this method is overrided.

        Parameters
        ----------
        params : dict


        Returns
        -------
        params : dict

        """

        return params

    def sample_mean(self):
        NotImplementedError

    def __mul__(self, other):
        return MultiplyDistribution(self, other)

    def __str__(self):
        return self.prob_text


class MultiplyDistribution(Distribution):
    """
    Multiply by given distributions, e.g, p(x,y|z) = p(x|z,y)p(y|z).
    In this class, it is checked if two distributions can be multiplied.

    p(x|z)p(z|y) -> Valid
    p(x|z)p(y|z) -> Valid
    p(x|z)p(y|a) -> Valid
    p(x|z)p(z|x) -> Invalid (recursive)
    p(x|z)p(x|y) -> Invalid (conflict)

    Parameters
    -------
    a : Tars.Distribution

    b : Tars.Distribution

    Examples
    --------
    >>> p_multi = MultipleDistribution([a, b])
    >>> p_multi = a * b
    """

    def __init__(self, a, b):
        if not (isinstance(a, Distribution) and isinstance(b, Distribution)):
            raise ValueError("Given inputs should be `Tars.Distribution`, got {} and {}.".format(type(a), type(b)))

        # Check parent-child relationship between two distributions.
        # If inherited variables (`_inh_var`) are exist (e.g. c in p(e|c)p(c|a,b)),
        # then p(e|c) is a child and p(c|a,b) is a parent, otherwise it is opposite.
        _vars_a_b = a.cond_var + b.var
        _vars_b_a = b.cond_var + a.var
        _inh_var_a_b = [var for var in set(_vars_a_b) if _vars_a_b.count(var) > 1]
        _inh_var_b_a = [var for var in set(_vars_b_a) if _vars_b_a.count(var) > 1]

        if len(_inh_var_a_b) > 0:
            _children = a
            _parents = b
            _inh_var = _inh_var_a_b

        elif len(_inh_var_b_a) > 0:
            _children = b
            _parents = a
            _inh_var = _inh_var_b_a

        else:
            _children = a
            _parents = b
            _inh_var = []

        # Check if variables of two distributions are "recursive" (e.g. p(x|z)p(z|x)).
        _check_recursive_vars = _children.var + _parents.cond_var
        if len(_check_recursive_vars) != len(set(_check_recursive_vars)):
            raise ValueError("Variables of two distributions, {} and {}, are recursive.".format(_children.prob_text,
                                                                                                _parents.prob_text))

        # Set variables.
        _var = _children.var + _parents.var
        if len(_var) != len(set(_var)):  # e.g. p(x|z)p(x|y)
            raise ValueError("Variables of two distributions, {} and {}, are conflicted.".format(_children.prob_text,
                                                                                                 _parents.prob_text))

        # Set conditional variables.
        _cond_var = _children.cond_var + _parents.cond_var
        _cond_var = sorted(set(_cond_var), key=_cond_var.index)

        # Delete inh_var in conditional variables.
        _cond_var = [var for var in _cond_var if var not in _inh_var]

        super().__init__(cond_var=_cond_var, var=_var)

        self._inh_var = _inh_var
        self._parents = _parents
        self._children = _children

    @property
    def inh_var(self):
        return self._inh_var

    @property
    def prob_factorized_text(self):
        return self._children.prob_factorized_text + self._parents.prob_text

    def _initialize_constant_params(self, **kwargs):
        pass

    def get_params(self, params):
        raise AttributeError

    def sample(self, x=None, shape=None, batch_size=1, return_all=True,
               reparam=True, **kwargs):
        """
        Sample variables of this distribution.
        If `cond_var` is not empty, we should set inputs as a dictionary format.

        Parameters
        ----------
        x : torch.Tensor, list, or dict
            Input variables.

        shape : tuple
            Shape of samples.
            If set, `batch_size` and `dim` are ignored.

        batch_size : int
            Batch size of samples. This is set to 1 by default.

        return_all : bool
            Choose whether the output contains input variables.

        reparam : bool
            Choose whether we sample variables with reparameterized trick.

        kwargs : dict

        Returns
        -------
        output : dict
            Samples of this distribution.
        """

        # input : dict
        # output : dict

        # sample from the parent distribution
        if x is None:
            if len(self._parents.cond_var) > 0:
                raise ValueError("You should set inputs.")

            parents_output = self._parents.sample(batch_size=batch_size)

        else:
            if batch_size == 1:
                batch_size = list(x.values())[0].shape[0]

            if list(x.values())[0].shape[0] != batch_size:
                raise ValueError("Invalid batch size")

            if set(list(x.keys())) != set(self.cond_var):
                raise ValueError("Input's keys are not valid.")

            if len(self._parents.cond_var) > 0:
                parents_input = get_dict_values(
                    x, self._parents.cond_var, return_dict=True)
                parents_output = self._parents.sample(
                    parents_input, return_all=False)
            else:
                parents_output = self._parents.sample(
                    batch_size=batch_size, return_all=False)

        # sample from the child distribution
        children_input_inh = get_dict_values(
            parents_output, self.inh_var, return_dict=True)
        if x is None:
            children_input = children_input_inh
        else:
            children_cond_exc_inh = list(
                set(self._children.cond_var)-set(self.inh_var))
            children_input = get_dict_values(
                x, children_cond_exc_inh, return_dict=True)
            children_input.update(children_input_inh)

        children_output = self._children.sample(
            children_input, return_all=False)

        output = parents_output
        output.update(children_output)

        if return_all and x:
            output.update(x)

        return output

    def sample_mean(self, x=None, batch_size=1, *args, **kwargs):
        # input : dict
        # output : dict

        # sample from the parent distribution
        if x is None:
            if len(self._parents.cond_var) > 0:
                raise ValueError("You should set inputs.")

            parents_output = self._parents.sample(batch_size=batch_size)

        else:
            if batch_size == 1:
                batch_size = list(x.values())[0].shape[0]

            if list(x.values())[0].shape[0] != batch_size:
                raise ValueError("Invalid batch size")

            if set(list(x.keys())) != set(self.cond_var):
                raise ValueError("Input's keys are not valid.")

            if len(self._parents.cond_var) > 0:
                parents_input = get_dict_values(
                    x, self._parents.cond_var, return_dict=True)
                parents_output = self._parents.sample(
                    parents_input, return_all=False)
            else:
                parents_output = self._parents.sample(
                    batch_size=batch_size, return_all=False)

        # sample from the child distribution
        children_input_inh = get_dict_values(
            parents_output, self.inh_var, return_dict=True)
        if x is None:
            children_input = children_input_inh
        else:
            children_cond_exc_inh = list(
                set(self._children.cond_var)-set(self.inh_var))
            children_input = get_dict_values(
                x, children_cond_exc_inh, return_dict=True)
            children_input.update(children_input_inh)

        output = self._children.sample_mean(children_input)
        return output

    def log_likelihood(self, x):
        """
        Estimate the log likelihood of this distribution from inputs formatted by a dictionary.

        Parameters
        ----------
        x : dict


        Returns
        -------
        log_like : torch.Tensor

        """

        parents_x = get_dict_values(
            x, self._parents.cond_var + self._parents.var,
            return_dict=True)
        children_x = get_dict_values(
            x, self._children.cond_var + self._children.var,
            return_dict=True)

        log_like = self._parents.log_likelihood(parents_x) +\
            self._children.log_likelihood(children_x)

        return log_like

    def forward(self, *args, **kwargs):
        NotImplementedError

    def __str__(self):
        return self.prob_text + " = " + self.prob_factorized_text


def mean_sum_samples(samples):
    dim = samples.dim()
    if dim == 4:
        return torch.mean(torch.sum(torch.sum(samples, dim=2), dim=2), dim=1)
    elif dim == 3:
        return torch.sum(torch.sum(samples, dim=-1), dim=-1)
    elif dim == 2:
        return torch.sum(samples, dim=-1)
    elif dim == 1:
        return samples
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % dim)

