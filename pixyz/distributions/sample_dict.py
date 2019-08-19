from collections import OrderedDict
from collections.abc import Container
from numbers import Number
from typing import Mapping
import sys
import torch


class ShapeDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], dict) and sys.version_info[0] < 3 or sys.version_info[1] < 7:
                raise ValueError('')
        super().__init__(*args, **kwargs)

    def shape_dims(self, shape_name):
        """
        Return a list of dim which are named as shape_name.

        Parameters
        ----------
        shape_name : str

        Returns
        -------
        list of int

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import ShapeDict
        >>> shape_dict = ShapeDict([('sample', [4, 6, 2]), ('batch', [8]), ('feature', [3, 2, 9])])
        >>> shape_dict.shape_dims('sample')
        [0, 1, 2]
        >>> shape_dict.shape_dims('batch')
        [3]
        >>> shape_dict.shape_dims('feature')
        [4, 5, 6]

        """
        start_dim = 0
        for shape_name2, shape in self.items():
            if shape_name == shape_name2:
                break
            start_dim += len(shape)
        return list(range(start_dim, start_dim + len(self[shape_name])))


class Sample:
    """
    Sampled result class. It has info of meaning of shapes.

    """
    def __init__(self, value, shape_dict=None):
        if shape_dict and not isinstance(shape_dict, ShapeDict):
            shape_dict = ShapeDict(shape_dict)
        self.value = value
        self.shape_dict = shape_dict if shape_dict else self._default_shape(value)

    def _default_shape(self, value):
        if isinstance(value, Number):
            return ShapeDict()
        if not isinstance(value, torch.Tensor):
            raise ValueError
        if value.dim() == 0:
            return ShapeDict()
        elif value.dim() == 1:
            return ShapeDict((('feature', [value.shape[0]]),))
        return ShapeDict((('batch', list(value.shape)[:1]),
                          ('feature', list(value.shape)[1:])))

    def detach(self):
        return Sample(self.value.detach(), self.shape_dict)

    def slice(self, index, shape_name='time'):
        if isinstance(index, int):
            index = [index]
        if shape_name not in self.shape_dict:
            return self
        shape_dict = self.shape_dict.copy()
        shape_dims = self.shape_dims(shape_name)
        value = self._slice_value(self.value, index, shape_dims)
        if all(not isinstance(item, slice) for item in index):
            del shape_dict[shape_name]
        else:
            sliced_shape = []
            for i, shape_dim in enumerate(shape_dims):
                if isinstance(index[i], int):
                    continue
                sliced_shape.append(len(range(*index[i].indices(self.value.shape[shape_dim]))))
            shape_dict[shape_name] = sliced_shape
        return Sample(value, shape_dict)

    def shape_dims(self, shape_name):
        return self.shape_dict.shape_dims(shape_name)

    def _slice_value(self, value, index, shape_dims):
        slices = [slice(None) if dim not in shape_dims else index[shape_dims.index(dim)]
                  for dim in range(value.dim())]
        return value[slices]

    @property
    def feature_shape(self):
        return self.shape_dict['feature']

    @property
    def n_batch(self):
        return self.shape_dict['batch'][0]

    def sum(self, shape_name):
        """
        Sum up over shape_name dims (inplace)

        Parameters
        ----------
        shape_name : str

        Returns
        -------
        torch.Tensor
        inplaced sum tensor

        """
        self.value = torch.sum(self.value, dim=self.shape_dims(shape_name))
        shape_dict = self.shape_dict.copy()
        del shape_dict[shape_name]
        self.shape_dict = shape_dict
        return self.value

    def __repr__(self):
        return f"{repr(self.value)} --(shape={repr(list(self.shape_dict.items()))})"


class SampleDict(Mapping):
    """
    Container class of sampled values. Each value has info of meaning of shapes.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal, Sample
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...             features_shape=[5], name="p")
    >>> sample_dict = p.sample(sample_shape=[3, 4])
    >>> sample_dict['x'].shape
    torch.Size([3, 4, 1, 5])
    >>> sample_dict.get_shape('x')
    ShapeDict([('sample', [3, 4]), ('batch', [1]), ('feature', [5])])
    >>> sample_dict = p.sample(batch_n=2)
    >>> sample_dict.get_shape('x')
    ShapeDict([('batch', [2]), ('feature', [5])])
    >>> sample = sample_dict.get_sample('x')
    >>> type(sample)
    <class 'pixyz.distributions.sample_dict.Sample'>
    >>> sample #doctest: +SKIP
    tensor([[-0.8378,  0.8379,  0.4045,  0.8837, -0.2362],
            [-1.9989,  0.8083, -1.1591, -1.5242,  0.4656]]) --(shape=[('batch', [2]), ('feature', [5])])
    >>> sample_dict.add('y', torch.zeros(4, 2, 3), shape_dict=(('sample', [4]), ('batch', [2]), ('feature', [3])))
    >>> sample_dict.get_shape('y')
    ShapeDict([('sample', [4]), ('batch', [2]), ('feature', [3])])
    >>> sample_dict.add('z', Sample(torch.zeros(4, 2, 3), (('sample', [4]), ('batch', [2]), ('feature', [3]))))
    >>> sample_dict.get_shape('z')
    ShapeDict([('sample', [4]), ('batch', [2]), ('feature', [3])])
    >>> sample_dict.n_batch('z')
    2
    >>> sample_dict.feature_shape('z')
    [3]
    >>> q = Normal(loc='x', scale=torch.tensor(1.), var=["y"], cond_var=["x"], name="q")
    >>> sample_dict = q.sample(p.sample(sample_shape=[2]), sample_shape=[3, 4], batch_n=6)
    >>> sample_dict.get_shape('y')
    ShapeDict([('sample', [3, 4, 2]), ('batch', [6]), ('feature', [5])])
    """
    def __init__(self, variables):
        """
        Initialize SampleDict from mapping of variable's name and the value of variable.

        Parameters
        ----------
        variables : Mapping of str and torch.Tensor or Mapping of str and Sample or SampleDict

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import SampleDict
        >>> sample_dict = SampleDict({'x': torch.zeros(2, 3, 4)})
        >>> sample_dict.get_shape('x')
        ShapeDict([('batch', [2]), ('feature', [3, 4])])
        >>> sample_dict = SampleDict({'x': Sample(torch.zeros(2, 3, 4),
        ...                                       shape_dict=(('feature', [2, 3, 4]),))})
        >>> sample_dict.get_shape('x')
        ShapeDict([('feature', [2, 3, 4])])
        >>> sample_dict = SampleDict({'x': Sample(torch.zeros(2, 3, 4),
        ...                                       shape_dict=(('time', [2]), ('batch', [3]), ('feature', [4])))})
        >>> sample_dict.get_shape('x')
        ShapeDict([('time', [2]), ('batch', [3]), ('feature', [4])])
        """
        if isinstance(variables, dict):
            variables = variables.items()
        elif isinstance(variables, SampleDict):
            variables = variables._dict.items()
        self._dict = {key: value if isinstance(value, Sample) else Sample(value) for key, value in variables}

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, var_name):
        return self._dict[var_name].value

    def __setitem__(self, var_name, value):
        self._dict[var_name] = Sample(value)

    def get_shape(self, var_name):
        return self._dict[var_name].shape_dict

    def get_sample(self, var_name):
        return self._dict[var_name]

    def items(self):
        return ((key, sample.value) for key, sample in self._dict.items())

    def keys(self):
        return self._dict.keys()

    def __contains__(self, key):
        return self._dict.__contains__(key)

    def __len__(self):
        return self._dict.__len__()

    def __eq__(self, other):
        if not isinstance(other, SampleDict):
            return False
        return self._dict.__eq__(other._dict)

    def __str__(self):
        return str(dict(self.items()))

    def __repr__(self):
        return self._dict.__repr__()

    def add(self, var_name, value, shape_dict=None):
        """
        Add key and value to SampleDict with shape_dict info.

        Parameters
        ----------
        var_name : str
        value : torch.Tensor or Sample
        shape_dict: Iterable of tuple or ShapeDict, optional
        """
        if shape_dict and not isinstance(shape_dict, ShapeDict):
            shape_dict = ShapeDict(shape_dict)
        if isinstance(value, Sample):
            self._dict[var_name] = value
        else:
            self._dict[var_name] = Sample(value, shape_dict)

    def update(self, variables):
        if isinstance(variables, dict):
            variables = SampleDict(variables)
        if not isinstance(variables, SampleDict):
            raise ValueError
        self._dict.update(variables._dict)

    def copy(self):
        return SampleDict(self._dict)

    def detach(self):
        """Return new SampleDict whose values are detached.

        Returns
        -------
        SampleDict
        """
        return SampleDict((var_name, sample.detach()) for var_name, sample in self._dict.items())

    def slice(self, index, shape_name='time'):
        """
        Return new SampleDict whose values are sliced by shape_name and index
        Parameters
        ----------
        index : int or list of int or list of slice
        shape_name : str

        Returns
        -------
        SampleDict

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import SampleDict, Sample
        >>> sample_dict = SampleDict({'x': Sample(torch.zeros(2, 3, 4),
        ...                                       shape_dict=(('time', [2]), ('batch', [3]), ('feature', [4])))})
        >>> sample_dict.get_shape('x')
        ShapeDict([('time', [2]), ('batch', [3]), ('feature', [4])])
        >>> sliced = sample_dict.slice(0)
        >>> sliced.get_shape('x')
        ShapeDict([('batch', [3]), ('feature', [4])])
        >>> sample_dict = SampleDict({'x': Sample(torch.zeros(2, 3, 4),
        ...                                       shape_dict=(('time', [2, 3]), ('feature', [4])))})
        >>> sliced = sample_dict.slice([1, slice(None)])
        >>> sliced.get_shape('x')
        ShapeDict([('time', [3]), ('feature', [4])])
        >>> sliced = sample_dict.slice([1, 2])
        >>> sliced.get_shape('x')
        ShapeDict([('feature', [4])])

        """
        return SampleDict((var_name, sample.slice(index, shape_name)) for var_name, sample in self._dict.items())

    def values(self):
        return (sample.value for sample in self._dict.values())

    def extract(self, var, return_dict=False):
        """Get values from `dicts` specified by `keys`.

        When `return_dict` is True, return values are in dictionary format.

        Parameters
        ----------
        var : list

        return_dict : bool

        Returns
        -------
        dict or list

        Examples
        --------
        >>> SampleDict({"a":1,"b":2,"c":3}).extract(["b"])
        [2]
        >>> SampleDict({"a":1,"b":2,"c":3}).extract(["b", "d"], return_dict=True)
        {'b': 2 --(shape=[])}
        """
        if not return_dict:
            return list(sample.value for var_name, sample in self._dict.items() if var_name in var)
        return SampleDict((var_name, sample) for var_name, sample in self._dict.items() if var_name in var)

    def dict_except_for_keys(self, var: Container):
        """Delete values from `dicts` specified by `keys`.

        Parameters
        ----------
        var : Container

        Returns
        -------
        new_dicts : SampleDict

        Examples
        --------
        >>> SampleDict({"a":1,"b":2,"c":3}).dict_except_for_keys(["b","d"])
        {'a': 1 --(shape=[]), 'c': 3 --(shape=[])}
        """
        return SampleDict((var_name, sample) for var_name, sample in self._dict.items() if var_name not in var)

    def dict_with_replaced_keys(self, replace_list_dict):
        """ Replace values in `dicts` according to `replace_list_dict`.

        Parameters
        ----------
        replace_list_dict : dict
            Dictionary.

        Returns
        -------
        replaced_dicts : SampleDict

        Examples
        --------
        >>> SampleDict({"a":1,"b":2,"c":3}).dict_with_replaced_keys({"a":"x","b":"y"})
        {'x': 1 --(shape=[]), 'y': 2 --(shape=[]), 'c': 3 --(shape=[])}
        >>> SampleDict({"a":1,"b":2,"c":3}).dict_with_replaced_keys({"a":"x","e":"y"})  # keys of `replace_list_dict`
        {'x': 1 --(shape=[]), 'b': 2 --(shape=[]), 'c': 3 --(shape=[])}
        """
        return SampleDict((replace_list_dict[var_name] if var_name in replace_list_dict else var_name, sample)
                          for var_name, sample in self._dict.items())

    def split_by_replace_keys(self, replace_list_dict):
        """ Replace values in `dicts` according to :attr:`replace_list_dict`.

        Replaced dict is splitted by :attr:`replaced_dict` and :attr:`remain_dict`.

        Parameters
        ----------
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
        >>> x_dict = SampleDict({'a': 0, 'b': 1})
        >>> print(x_dict.split_by_replace_keys(replace_list_dict))
        ({'loc': 0 --(shape=[])}, {'b': 1 --(shape=[])})

        """
        replaced_sample = SampleDict((replace_list_dict[var_name], sample)
                                     for var_name, sample in self._dict.items() if var_name in replace_list_dict)

        remain_sample = SampleDict((var_name, sample)
                                   for var_name, sample in self._dict.items() if var_name not in replace_list_dict)

        return replaced_sample, remain_sample

    @property
    def max_shape(self):
        """
        Get max shape which all other values can be broadcasted to.

        Returns
        -------
        ShapeDict

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import SampleDict
        >>> sample_dict = SampleDict({'x': Sample(torch.zeros(2, 3, 4),
        ...                                       shape_dict=(('time', [2]), ('batch', [3]), ('feature', [4]))),
        ...                           'mu': Sample(torch.zeros(4),
        ...                                       shape_dict=(('feature', [4]),)),
        ...                           'sigma': Sample(torch.zeros(1, 3, 4),
        ...                                       shape_dict=(('time', [1]), ('batch', [3]), ('feature', [4]))),
        ...                           })
        >>> sample_dict.max_shape
        ShapeDict([('time', [2]), ('batch', [3]), ('feature', [4])])
        """
        result = []
        for var_name, sample in self._dict.items():
            for i, (shape_name, shape) in enumerate(reversed(sample.shape_dict.items())):
                if len(result) <= i:
                    result.append((shape_name, shape))
                    continue
                if shape_name != result[i][0]:
                    raise ValueError
                if len(result[i][1]) == 0:
                    result[i] = (result[i][0], shape)
                elif len(result[i][1]) != len(shape):
                    raise ValueError
        return ShapeDict(reversed(result))

    def feature_shape(self, var_name):
        return self._dict[var_name].feature_shape

    def n_batch(self, var_name):
        return self._dict[var_name].n_batch
