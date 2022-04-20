import functools
import torch
import sympy
from IPython.display import Math
import pixyz

_EPSILON = 1e-07
_CACHE_MAXSIZE = 2 * 10


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


def set_cache_maxsize(cache_maxsize):
    """Set a `cache_maxsize` parameter.

    Parameters
    ----------
    cache_maxsize : int

    Returns
    -------

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._CACHE_MAXSIZE', 100):
    ...     set_cache_maxsize(100)
    ...     cache_maxsize()
    100
    """
    global _CACHE_MAXSIZE
    _CACHE_MAXSIZE = cache_maxsize


def cache_maxsize():
    """Get a `cache_maxsize` parameter.

    Returns
    -------
    int

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._CACHE_MAXSIZE', 100):
    ...     cache_maxsize()
    100
    """
    return _CACHE_MAXSIZE


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


# immutable dict class
class FrozenSampleDict:
    def __init__(self, dict_):
        self.dict = dict_

    def __hash__(self):
        hashes = [(hash(key), hash(value)) for key, value in self.dict.items()]
        return hash(tuple(hashes))

    def __eq__(self, other):
        class EqTensor:
            def __init__(self, tensor):
                self.tensor = tensor

            def __eq__(self, other):
                if not torch.is_tensor(self.tensor):
                    return self.tensor == other.tensor
                return torch.all(self.tensor.eq(other.tensor))
        return {key: EqTensor(value) for key, value in self.dict.items()} ==\
               {key: EqTensor(value) for key, value in other.dict.items()}


def lru_cache_for_sample_dict():
    """
    Memoize the calculation result linked to the argument of sample dict.
    Note that dictionary arguments of the target function must be sample dict.

    Returns
    -------
    decorator function

    Examples
    --------
    >>> import time
    >>> import torch.nn as nn
    >>> import pixyz.utils as utils
    >>> utils.set_cache_maxsize(2)
    >>> import pixyz.distributions as pd
    >>> class LongEncoder(pd.Normal):
    ...     def __init__(self):
    ...         super().__init__(var=['x'], cond_var=['y'])
    ...         self.nn = nn.Sequential(*(nn.Linear(1,1) for i in range(10000)))
    ...     def forward(self, y):
    ...         return {'loc': self.nn(y), 'scale': torch.ones(1,1)}
    ...     @lru_cache_for_sample_dict()
    ...     def get_params(self, params_dict={}, **kwargs):
    ...         return super().get_params(params_dict, **kwargs)
    >>> def measure_time(func):
    ...     start = time.time()
    ...     func()
    ...     elapsed_time = time.time() - start
    ...     return elapsed_time
    >>> le = LongEncoder()
    >>> y = torch.ones(1, 1)
    >>> t_sample1 = measure_time(lambda:le.sample({'y': y}))
    >>> print ("sample1:{0}".format(t_sample1) + "[sec]") # doctest: +SKIP
    >>> t_log_prob = measure_time(lambda:le.get_log_prob({'x': y, 'y': y}))
    >>> print ("log_prob:{0}".format(t_log_prob) + "[sec]") # doctest: +SKIP
    >>> t_sample2 = measure_time(lambda:le.sample({'y': y}))
    >>> print ("sample2:{0}".format(t_sample2) + "[sec]") # doctest: +SKIP
    >>> assert t_sample1 > t_sample2, "processing time increases: {0}".format(t_sample2 - t_sample1)
    """
    maxsize = cache_maxsize()
    raw_decorating_function = functools.lru_cache(maxsize=maxsize, typed=False)

    def decorating_function(user_function):
        def wrapped_user_function(sender, *args, **kwargs):
            new_args = list(args)
            new_kwargs = dict(kwargs)
            for i in range(len(args)):
                if isinstance(args[i], FrozenSampleDict):
                    new_args[i] = args[i].dict
            for key in kwargs.keys():
                if isinstance(kwargs[key], FrozenSampleDict):
                    new_kwargs[key] = kwargs[key].dict
            return user_function(sender, *new_args, **new_kwargs)

        def frozen(wrapper):
            def frozen_wrapper(sender, *args, **kwargs):
                if len(args) == 0 and len(kwargs) == 0:
                    result = wrapper(sender)
                    return result
                new_args = list(args)
                new_kwargs = dict(kwargs)
                for i in range(len(args)):
                    if isinstance(args[i], list):
                        new_args[i] = tuple(args[i])
                    elif isinstance(args[i], dict):
                        new_args[i] = FrozenSampleDict(args[i])
                for key in kwargs.keys():
                    if isinstance(kwargs[key], list):
                        new_kwargs[key] = tuple(kwargs[key])
                    elif isinstance(kwargs[key], dict):
                        new_kwargs[key] = FrozenSampleDict(kwargs[key])
                result = wrapper(sender, *new_args, **new_kwargs)
                return result
            return frozen_wrapper
        return frozen(raw_decorating_function(wrapped_user_function))
    return decorating_function


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


def sum_samples(samples, sum_dims=None):
    """Sum a given sample across the axes.

    Parameters
    ----------
    samples : torch.Tensor
        Input sample.
    sum_dims : torch.Size or list of int or None
        Dimensions to reduce. If it is None, all dimensions are summed except for the first dimension.

    Returns
    -------
    torch.Tensor
        Sumed sample.


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
    if sum_dims is not None:
        if len(sum_dims) == 0:
            return samples
        return torch.sum(samples, dim=sum_dims)

    dim = samples.dim()
    if dim == 1:
        return samples
    dim_list = list(torch.arange(samples.dim()))
    samples = torch.sum(samples, dim=dim_list[1:])
    return samples


def print_latex(obj):
    """Print formulas in latex format.

    Parameters
    ----------
    obj : pixyz.distributions.distributions.Distribution, pixyz.losses.losses.Loss or pixyz.models.model.Model.

    """

    if isinstance(obj, pixyz.distributions.distributions.Distribution):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.distributions.distributions.DistGraph):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.losses.losses.Loss):
        latex_text = obj.loss_text
    elif isinstance(obj, pixyz.models.model.Model):
        latex_text = obj.loss_cls.loss_text

    return Math(latex_text)


def convert_latex_name(name):
    return sympy.latex(sympy.Symbol(name))
