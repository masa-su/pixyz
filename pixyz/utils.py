from collections import OrderedDict, Mapping, Iterable
import torch
import sympy
from IPython.display import Math
import pixyz

_EPSILON = 1e-07


def set_epsilon(eps):
    """Set a `epsilon` parameter.

    Parameters
    ----------
    eps : int or float

    Returns
    -------

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._EPSILON', 1e-07):
    ...     set_epsilon(1e-06)
    ...     epsilon()
    1e-06
    """
    global _EPSILON
    _EPSILON = eps


def epsilon():
    """Get a `epsilon` parameter.

    Returns
    -------
    int or float

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._EPSILON', 1e-07):
    ...     epsilon()
    1e-07
    """
    return _EPSILON


class Sample:
    def __init__(self, value, shape_dict=None):
        if shape_dict and not isinstance(shape_dict, OrderedDict):
            raise ValueError
        self.value = value
        self.shape_dict = shape_dict if shape_dict else self._default_shape(value)

    def _default_shape(self, value):
        return OrderedDict((('batch', list(value.shape)[:0]), ('feature', list(value.shape)[1:])))

    def detach(self):
        return Sample(self.value.detach(), self.shape_dict)

    def slice(self, index, shape_name='time'):
        shape_dict = self.shape_dict.copy()
        shape_dims = self._get_shape_dims(shape_dict, shape_name)
        value = self._slice_value(self.value, index, shape_dims)
        del shape_dict[shape_name]
        return Sample(value, shape_dict)

    def _get_shape_dims(self, shape_dict: OrderedDict, shape_name):
        start_dim = 0
        for shape_name2, shape in shape_dict.items():
            if shape_name == shape_name2:
                break
            start_dim += len(shape)
        return range(start_dim, start_dim + len(shape_dict[shape_name]))

    def _slice_value(self, value, index, shape_dims):
        slices = [slice(None, None, None) if dim not in shape_dims else index[shape_dims.index(dim)]
                  for dim in range(value.dim())]
        return value[slices]

    @property
    def feature_shape(self):
        return self.shape_dict['feature']

    @property
    def n_batch(self):
        return self.shape_dict['batch'][0]


class Samples:
    def __init__(self, variables):
        if isinstance(variables, dict):
            variables = variables.items()
        self._dict = {key: value if isinstance(value, Sample) else Sample(value) for key, value in variables}

    def __getitem__(self, var_name):
        return self._dict[var_name].value

    def get_shape(self, var_name):
        return self._dict[var_name].shape

    def items(self):
        return ((key, sample.value) for key, sample in self._dict.items())

    def add(self, var_name, value, shape_dict=None):
        if shape_dict and not isinstance(shape_dict, OrderedDict):
            if isinstance(shape_dict, Mapping) or isinstance(shape_dict, Iterable):
                shape_dict = OrderedDict(shape_dict)
            else:
                raise ValueError
        self._dict[var_name] = Sample(value, shape_dict)

    def update(self, variables):
        if isinstance(variables, dict):
            variables = Samples(variables)
        if not isinstance(variables, Samples):
            raise ValueError
        self._dict.update(variables._dict)

    def copy(self):
        return Samples(self._dict)

    def detach(self):
        return Samples((var_name, sample.detach()) for var_name, sample in self._dict.items())

    def slice(self, index, shape_name='time'):
        return Samples((var_name, sample.slice(index, shape_name)) for var_name, sample in self._dict.items())

    def get_values(self, var, return_tensors=True):
        if return_tensors:
            return (sample.value for sample in self._dict.values())
        return Samples((var_name, sample) for var_name, sample in self._dict.items() if var_name in var)

    def delete_values(self, var):
        return Samples((var_name, sample) for var_name, sample in self._dict.items() if var_name not in var)

    def replace_keys(self, replace_list_dict):
        return Samples((replace_list_dict[var_name] if var_name in replace_list_dict else var_name, sample)
                       for var_name, sample in self._dict.items())

    def replace_keys_split(self, replace_list_dict):
        replaced_sample = Samples((replace_list_dict[var_name], sample)
                                  for var_name, sample in self._dict.items() if var_name in replace_list_dict)

        remain_sample = Samples((var_name, sample)
                                for var_name, sample in self._dict.items() if var_name not in replace_list_dict)

        return replaced_sample, remain_sample

    def feature_shape(self, var_name):
        return self._dict[var_name].feature_shape

    def n_batch(self, var_name):
        return self._dict[var_name].n_batch


def get_dict_values(dicts, keys, return_dict=False):
    """Get values from `dicts` specified by `keys`.

    When `return_dict` is True, return values are in dictionary format.

    Parameters
    ----------
    dicts : dict

    keys : list

    return_dict : bool

    Returns
    -------
    dict or list

    Examples
    --------
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b"])
    [2]
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b", "d"], True)
    {'b': 2}
    """
    new_dicts = dict((key, dicts[key]) for key in keys if key in list(dicts.keys()))
    if return_dict is False:
        return list(new_dicts.values())

    return new_dicts


def delete_dict_values(dicts, keys):
    """Delete values from `dicts` specified by `keys`.

    Parameters
    ----------
    dicts : dict

    keys : list

    Returns
    -------
    new_dicts : dict

    Examples
    --------
    >>> delete_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    {'a': 1, 'c': 3}
    """
    new_dicts = dict((key, value) for key, value in dicts.items() if key not in keys)
    return new_dicts


def detach_dict(dicts):
    """Detach all values in `dicts`.

    Parameters
    ----------
    dicts : dict

    Returns
    -------
    dict
    """
    return {k: v.detach() for k, v in dicts.items()}


def replace_dict_keys(dicts, replace_list_dict):
    """ Replace values in `dicts` according to `replace_list_dict`.

    Parameters
    ----------
    dicts : dict
        Dictionary.
    replace_list_dict : dict
        Dictionary.

    Returns
    -------
    replaced_dicts : dict
        Dictionary.

    Examples
    --------
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":"x","b":"y"})
    {'x': 1, 'y': 2, 'c': 3}
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":"x","e":"y"})  # keys of `replace_list_dict`
    {'x': 1, 'b': 2, 'c': 3}
    """
    replaced_dicts = dict([(replace_list_dict[key], value) if key in list(replace_list_dict.keys())
                           else (key, value) for key, value in dicts.items()])

    return replaced_dicts


def replace_dict_keys_split(dicts, replace_list_dict):
    """ Replace values in `dicts` according to :attr:`replace_list_dict`.

    Replaced dict is splitted by :attr:`replaced_dict` and :attr:`remain_dict`.

    Parameters
    ----------
    dicts : dict
        Dictionary.
    replace_list_dict : dict
        Dictionary.

    Returns
    -------
    replaced_dict : dict
        Dictionary.
    remain_dict : dict
        Dictionary.

    Examples
    --------
    >>> replace_list_dict = {'a': 'loc'}
    >>> x_dict = {'a': 0, 'b': 1}
    >>> print(replace_dict_keys_split(x_dict, replace_list_dict))
    ({'loc': 0}, {'b': 1})

    """
    replaced_dict = {replace_list_dict[key]: value for key, value in dicts.items()
                     if key in list(replace_list_dict.keys())}

    remain_dict = {key: value for key, value in dicts.items()
                   if key not in list(replace_list_dict.keys())}

    return replaced_dict, remain_dict


def tolist(a):
    """Convert a given input to the dictionary format.

    Parameters
    ----------
    a : list or other

    Returns
    -------
    list

    Examples
    --------
    >>> tolist(2)
    [2]
    >>> tolist([1, 2])
    [1, 2]
    >>> tolist([])
    []
    """
    if type(a) is list:
        return a
    return [a]


def sum_samples(samples):
    """Sum a given sample across the axes.

    Parameters
    ----------
    samples : torch.Tensor
        Input sample. The number of this axes is assumed to be 4 or less.

    Returns
    -------
    torch.Tensor
        Sum over all axes except the first axis.


    Examples
    --------
    >>> a = torch.ones([2])
    >>> sum_samples(a).size()
    torch.Size([2])
    >>> a = torch.ones([2, 3])
    >>> sum_samples(a).size()
    torch.Size([2])
    >>> a = torch.ones([2, 3, 4])
    >>> sum_samples(a).size()
    torch.Size([2])
    """

    dim = samples.dim()
    if dim == 1:
        return samples
    elif dim <= 4:
        dim_list = list(torch.arange(samples.dim()))
        samples = torch.sum(samples, dim=dim_list[1:])
        return samples
    raise ValueError("The number of sample axes must be any of 1, 2, 3, or 4, "
                     "got %s." % dim)


def print_latex(obj):
    """Print formulas in latex format.

    Parameters
    ----------
    obj : pixyz.distributions.distributions.Distribution, pixyz.losses.losses.Loss or pixyz.models.model.Model.

    """

    if isinstance(obj, pixyz.distributions.distributions.Distribution):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.losses.losses.Loss):
        latex_text = obj.loss_text
    elif isinstance(obj, pixyz.models.model.Model):
        latex_text = obj.loss_cls.loss_text

    return Math(latex_text)


def convert_latex_name(name):
    return sympy.latex(sympy.Symbol(name))
