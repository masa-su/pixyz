from __future__ import print_function
import torch
import numbers
import re
from torch import nn
from copy import deepcopy

from ..utils import get_dict_values, replace_dict_keys, delete_dict_values, tolist


class Distribution(nn.Module):
    """
    Distribution class. In pixyz, all distributions are required to inherit this class.

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

    def __init__(self, cond_var=[], var=["x"], name="p", dim=1):
        super().__init__()
        _vars = cond_var + var
        if len(_vars) != len(set(_vars)):
            raise ValueError("There are conflicted variables.")

        self._cond_var = cond_var
        self._var = var
        self.dim = dim
        self._name = name

        self._prob_text = None
        self._prob_factorized_text = None

    @property
    def distribution_name(self):
        return None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if type(name) is str:
            self._name = name
            return

        raise ValueError("Name of the distribution class must be set as a string type.")

    @property
    def var(self):
        return self._var

    @property
    def cond_var(self):
        return self._cond_var

    @property
    def input_var(self):
        """
        Normally, `input_var` has same values as `cond_var`.
        """
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
            var = self.input_var

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

    def get_params(self, params_dict):
        """
        This method aims to get parameters of this distributions from constant parameters set in
        initialization and outputs of DNNs.

        Parameters
        ----------
        params_dict : dict

        Returns
        -------
        output_dict : dict

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

        NotImplementedError

    def sample(self, x={}, shape=None, batch_size=1, return_all=True,
               reparam=False):
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

        Returns
        -------
        output : dict
            Samples of this distribution.
        """

        NotImplementedError

    def log_likelihood(self, x_dict):
        """
        Estimate the log likelihood of this distribution from inputs formatted by a dictionary.

        Parameters
        ----------
        x_dict : dict


        Returns
        -------
        log_like : torch.Tensor

        """

        NotImplementedError

    def forward(self, *args, **kwargs):
        """
        When this class is inherited by DNNs, it is also intended that this method is overrided.

        Parameters
        ----------
        params : dict


        Returns
        -------
        params : dict

        """

        NotImplementedError

    def sample_mean(self, x):
        NotImplementedError

    def replace_var(self, **replace_dict):
        return ReplaceVarDistribution(self, replace_dict)

    def marginalize_var(self, marginalize_list):
        marginalize_list = tolist(marginalize_list)
        return MarginalizeVarDistribution(self, marginalize_list)

    def __mul__(self, other):
        return MultiplyDistribution(self, other)

    def __str__(self):
        # Distribution
        if self.prob_factorized_text == self.prob_text:
            prob_text = "{} ({})".format(self.prob_text, self.distribution_name)
        else:
            prob_text = "{} = {}".format(self.prob_text, self.prob_factorized_text)
        text = "Distribution:\n  {}\n".format(prob_text)

        # Network architecture (`repr`)
        network_text = self.__repr__()
        network_text = re.sub('^', ' ' * 2, str(network_text), flags=re.MULTILINE)
        text += "Network architecture:\n{}".format(network_text)
        return text


class DistributionBase(Distribution):

    def __init__(self, cond_var=[], var=["x"], name="p", dim=1, **kwargs):
        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim)

        self._set_constant_params(**kwargs)

    def _set_constant_params(self, **params_dict):
        """
        Format constant parameters of this distribution.

        Parameters
        ----------
        params_dict : dict
            Constant parameters of this distribution set at initialization.
            If the values of these dictionaries contain parameters which are named as strings, which means that
            these parameters are set as "variables", the correspondences between these values and the true name of
            these parameters are stored as a dictionary format (`replace_params_dict`).
        """

        self.replace_params_dict = {}
        self.constant_params_dict = {}

        for key in params_dict.keys():
            if type(params_dict[key]) is str:
                if params_dict[key] in self._cond_var:
                    self.replace_params_dict[params_dict[key]] = key
                else:
                    raise ValueError
            elif isinstance(params_dict[key], numbers.Number) or isinstance(params_dict[key], torch.Tensor):
                self.constant_params_dict[key] = params_dict[key]
            else:
                raise ValueError

    def _set_distribution(self, x={}):
        """
        Require self.params_keys and self.DistributionTorch

        Parameters
        ----------
        x : dict

        Returns
        -------

        """

        params = self.get_params(x)
        if set(self.params_keys) != set(params.keys()):
            raise ValueError

        self.dist = self.DistributionTorch(**params)

    def _get_sample(self, reparam=False,
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
                print("We can not use the reparameterization trick "
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

    def _replace_vars_to_params(self, vars_dict, replace_dict):
        """
        Replace variables in keys of a input dictionary to parameters of this distribution according to
        these correspondences which is formatted in a dictionary and set in `_initialize_constant_params`.

        Parameters
        ----------
        vars_dict : dict

        replace_dict : dict

        Returns
        -------
        params_dict : dict

        vars_dict : dict

        Examples
        --------
        >> replace_dict
        >> > {"a": "loc"}
        >> x = {"a": 0, "b": 1}
        >> distribution._replace_vars_to_params(x, replace_dict)
        >> > {"loc": 0}, {"b": 1}
        """

        params_dict = {replace_dict[key]: value for key, value in vars_dict.items()
                       if key in list(replace_dict.keys())}

        vars_dict = {key: value for key, value in vars_dict.items()
                     if key not in list(replace_dict.keys())}

        return params_dict, vars_dict

    def get_params(self, params_dict):
        params_dict, vars_dict = self._replace_vars_to_params(params_dict, self.replace_params_dict)
        output_dict = self.forward(**vars_dict)

        # append constant_params to dict
        output_dict.update(params_dict)
        output_dict.update(self.constant_params_dict)

        return output_dict

    def sample(self, x={}, shape=None, batch_size=1, return_all=True,
               reparam=False):

        if len(x) == 0:  # unconditioned
            if len(self.input_var) != 0:
                raise ValueError("You should set inputs or parameters")

            if shape:
                sample_shape = shape
            else:
                sample_shape = (batch_size, self.dim)

            self._set_distribution()
            output_dict = self._get_sample(reparam=reparam,
                                           sample_shape=sample_shape)

        else:  # conditioned
            x_dict = self._check_input(x)
            self._set_distribution(x_dict)
            output_dict = self._get_sample(reparam=reparam)

            if return_all:
                output_dict.update(x_dict)

        return output_dict

    def log_likelihood(self, x_dict):

        if not set(list(x_dict.keys())) >= set(self._cond_var + self._var):
            raise ValueError("Input's keys are not valid.")

        _x_dict = get_dict_values(x_dict, self._cond_var, True)
        self._set_distribution(_x_dict)

        log_like = self._get_log_like(x_dict)
        log_like = sum_samples(log_like)
        return log_like

    def forward(self, **params):
        return params


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
    a : pixyz.Distribution

    b : pixyz.Distribution

    Examples
    --------
    >>> p_multi = MultipleDistribution([a, b])
    >>> p_multi = a * b
    """

    def __init__(self, a, b):
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

        self._inh_var = _inh_var
        self._parent = _parent
        self._child = _child

        # Set input_var (it might be different from cond_var if either a and b contain data distributions.)
        _input_var = [var for var in self._child.input_var if var not in _inh_var]
        _input_var += self._parent.input_var
        self._input_var = sorted(set(_input_var), key=_input_var.index)

    @property
    def inh_var(self):
        return self._inh_var

    @property
    def input_var(self):
        return self._input_var

    @property
    def prob_factorized_text(self):
        return self._child.prob_factorized_text + self._parent.prob_factorized_text

    def sample(self, x={}, shape=None, batch_size=1, return_all=True,
               reparam=False):

        x = get_dict_values(x, self._input_var, return_dict=True)

        # sample from the parent distribution
        parents_input = get_dict_values(x, self._parent.input_var, return_dict=True)
        parents_output = self._parent.sample(x=parents_input,
                                             shape=shape,
                                             batch_size=batch_size,
                                             return_all=False, reparam=reparam)

        # sample from the child distribution
        children_inh_input = get_dict_values(parents_output, self.inh_var, return_dict=True)
        children_input_exc_inh_var = list(set(self._child.input_var)-set(self.inh_var))
        children_input = get_dict_values(x, children_input_exc_inh_var, return_dict=True)
        children_input.update(children_inh_input)

        children_output = self._child.sample(x=children_input,
                                             shape=shape,
                                             batch_size=batch_size,
                                             return_all=False, reparam=reparam)

        output = parents_output
        output.update(children_output)

        if return_all:
            output.update(x)

        return output

    def log_likelihood(self, x):
        parents_x = get_dict_values(x, self._parent.cond_var + self._parent.var, return_dict=True)
        children_x = get_dict_values(x, self._child.cond_var + self._child.var, return_dict=True)
        log_like = self._parent.log_likelihood(parents_x) + self._child.log_likelihood(children_x)

        return log_like

    def __repr__(self):
        if isinstance(self._parent, MultiplyDistribution):
            text = self._parent.__repr__()
        else:
            text = "{} ({}): {}".format(self._parent.prob_text, self._parent.distribution_name, self._parent.__repr__())
        text += "\n"

        if isinstance(self._child, MultiplyDistribution):
            text += self._child.__repr__()
        else:
            text += "{} ({}): {}".format(self._child.prob_text, self._child.distribution_name, self._child.__repr__())
        return text


class ReplaceVarDistribution(Distribution):
    """
    Replace names of variables in Distribution.

    Attributes
    ----------
    a : pixyz.Distribution (not pixyz.MultiplyDistribution)

    replace_dict : dict
    """

    def __init__(self, a, replace_dict):

        if not isinstance(a, Distribution):
            raise ValueError("Given input should be `pixyz.Distribution`, got {}.".format(type(a)))

        if isinstance(a, MultiplyDistribution):
            raise ValueError("`pixyz.MultiplyDistribution` is not supported for now.")

        if isinstance(a, MarginalizeVarDistribution):
            raise ValueError("`pixyz.MarginalizeVarDistribution` is not supported for now.")

        _cond_var = deepcopy(a.cond_var)
        _var = deepcopy(a.var)
        all_vars = _cond_var + _var

        if not (set(replace_dict.keys()) <= set(all_vars)):
            raise ValueError

        _replace_inv_cond_var_dict = {replace_dict[var]: var for var in _cond_var if var in replace_dict.keys()}
        _replace_inv_dict = {value: key for key, value in replace_dict.items()}

        self._replace_inv_cond_var_dict = _replace_inv_cond_var_dict
        self._replace_inv_dict = _replace_inv_dict
        self._replace_dict = replace_dict

        _cond_var = [replace_dict[var] if var in replace_dict.keys() else var for var in _cond_var]
        _var = [replace_dict[var] if var in replace_dict.keys() else var for var in _var]
        super().__init__(cond_var=_cond_var, var=_var, name=a.name, dim=a.dim)

        self._a = a
        _input_var = [replace_dict[var] if var in replace_dict.keys() else var for var in a.input_var]
        self._input_var = _input_var

    def forward(self, *args, **kwargs):
        return self._a.forward(*args, **kwargs)

    def get_params(self, params_dict):
        params_dict = replace_dict_keys(params_dict, self._replace_inv_cond_var_dict)
        return self._a.get_params(params_dict)

    def sample(self, x={}, shape=None, batch_size=1, return_all=True, reparam=False):
        x = replace_dict_keys(x, self._replace_inv_cond_var_dict)

        output_dict = self._a.sample(x, shape, batch_size, return_all, reparam)
        output_dict = replace_dict_keys(output_dict, self._replace_dict)

        return output_dict

    def log_likelihood(self, x):
        x = replace_dict_keys(x, self._replace_inv_dict)

        return self._a.log_likelihood(x)

    def sample_mean(self, x):
        x = replace_dict_keys(x, self._replace_inv_cond_var_dict)
        return self._a.sample_mean(x)

    @property
    def input_var(self):
        return self._input_var

    @property
    def distribution_name(self):
        return self._a.distribution_name

    def __repr__(self):
        return self._a.__repr__()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self._a.__getattribute__(item)


class MarginalizeVarDistribution(Distribution):
    """
    Marginalize variables in Distribution.
    p(x) = ∫p(x,z)dz

    Attributes
    ----------
    a : pixyz.Distribution (not pixyz.DistributionBase)

    marginalize_list : list
    """

    def __init__(self, a, marginalize_list):

        marginalize_list = tolist(marginalize_list)

        if not isinstance(a, Distribution):
            raise ValueError("Given input should be `pixyz.Distribution`, got {}.".format(type(a)))

        if isinstance(a, DistributionBase):
            raise ValueError("`pixyz.DistributionBase` cannot marginalize its variables for now.")

        _var = deepcopy(a.var)
        _cond_var = deepcopy(a.cond_var)

        if not((set(marginalize_list)) < set(_var)):
            raise ValueError()

        if not((set(marginalize_list)).isdisjoint(set(a.input_var))):
            raise ValueError()

        if len(marginalize_list) == 0:
            raise ValueError("Length of `marginalize_list` should be more than zero.")

        _var = [var for var in _var if var not in marginalize_list]

        super().__init__(cond_var=_cond_var, var=_var, name=a.name, dim=a.dim)
        self._a = a
        self._marginalize_list = marginalize_list

    def forward(self, *args, **kwargs):
        return self._a.forward(*args, **kwargs)

    def get_params(self, params_dict):
        return self._a.get_params(params_dict)

    def sample(self, x={}, shape=None, batch_size=1, return_all=True, reparam=False):
        output_dict = self._a.sample(x, shape, batch_size, False, reparam)
        output_dict = delete_dict_values(output_dict, self._marginalize_list)

        return output_dict

    def log_likelihood(self, x):
        NotImplementedError

    def sample_mean(self, x):
        return self._a.sample_mean(x)

    @property
    def input_var(self):
        return self._a.input_var

    @property
    def distribution_name(self):
        return self._a.distribution_name

    @property
    def prob_factorized_text(self):
        integral_symbol = len(self._marginalize_list) * "∫"
        integral_variables = ["d"+str(var) for var in self._marginalize_list]
        integral_variables = "".join(integral_variables)

        return "{}{}{}".format(integral_symbol, self._a.prob_factorized_text, integral_variables)

    def __repr__(self):
        return self._a.__repr__()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self._a.__getattribute__(item)


def sum_samples(samples):
    dim = samples.dim()

    if (dim >= 2) and (dim <= 4):
        for _ in range(1, dim):
            samples = torch.sum(samples, dim=-1)
        return samples
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % dim)

