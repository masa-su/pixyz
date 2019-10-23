from numbers import Number
from collections.abc import Iterable
from typing import Dict, List
from itertools import chain
import torch


class SampleDict(dict):
    """
    Container class of sampled values. Container has info of sample shape (& features shape).

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...             features_shape=[5], name="p")
    >>> sample_dict = p.sample(sample_shape=[3, 4])
    >>> sample_dict['x'].shape
    torch.Size([3, 4, 5])
    >>> sample_dict.features_shape('x')
    torch.Size([5])
    >>> sample_dict.sample_shape
    torch.Size([3, 4])
    >>> sample_dict.n_batch('x')
    4
    >>> sample_dict['y'] = torch.zeros(3, 4, 3, 2)
    >>> sample_dict.features_shape('y')
    torch.Size([3, 2])
    >>> sample_dict['z'] = torch.zeros(1, 4, 2, 3)
    >>> sample_dict.features_shape('z')
    torch.Size([2, 3])
    >>> sample_dict.add('w', torch.zeros(4, 2, 3), sample_shape=(4,))
    >>> sample_dict.features_shape('w')
    torch.Size([2, 3])
    >>> sample_dict['w'].shape
    torch.Size([1, 4, 2, 3])
    >>> q = Normal(loc='x', scale=torch.tensor(1.), var=["y"], cond_var=["x"], name="q")
    >>> sample_dict = q.sample(p.sample(sample_shape=[2]), sample_shape=[3, 4, 6])
    >>> sample_dict['y'].shape
    torch.Size([3, 4, 6, 2, 5])
    >>> sample_dict.sample_shape
    torch.Size([3, 4, 6, 2])
    """
    def __init__(self, variables=None, sample_shape=(), **kwargs):
        """
        Initialize SampleDict from mapping of variable's name and the value of variable.

        Parameters
        ----------
        variables : Mapping of str and torch.Tensor or Mapping of str and Sample or SampleDict
        sample_shape : Iterable or torch.Size

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import SampleDict
        >>> sample_dict = SampleDict({'x': torch.zeros(2, 3, 4)})
        >>> sample_dict.sample_shape
        torch.Size([])
        >>> sample_dict.features_shape('x')
        torch.Size([2, 3, 4])
        >>> sample_dict = SampleDict({'x': torch.zeros(2, 3, 4)}, sample_shape=(2, 3))
        >>> sample_dict.features_shape('x')
        torch.Size([4])
        """
        super().__init__()

        if isinstance(variables, SampleDict):
            self._sample_shape = variables._sample_shape
        else:
            self._sample_shape = torch.Size(sample_shape)
        if variables is None:
            variables = ()
        elif isinstance(variables, dict):
            variables = variables.items()
        for key, value in chain(variables, kwargs.items()):
            self[key] = value

    @staticmethod
    def is_broadcastable_to(from_shape, to_shape):
        for a, b in zip(from_shape[::-1], to_shape[::-1]):
            if a == 1 or a == b:
                pass
            else:
                return False
        return True

    def __setitem__(self, var_name, value):
        value = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float)
        if not self.is_broadcastable_to(value.shape[:len(self.sample_shape)], self.sample_shape):
            raise ValueError(f"the sample shape of value does not match. var_name: {var_name},"
                             f" expected: {self.sample_shape}, actual: {value.shape[:len(self.sample_shape)]}")
        super().__setitem__(var_name, value)

    def add(self, var_name, value, sample_shape):
        self.update(SampleDict({var_name: value}, sample_shape))

    @staticmethod
    def _check_dict_type(target):
        if not isinstance(target, SampleDict):
            raise ValueError(f'sample_dict is downgraded to {type(target)}.')

    @property
    def sample_dims(self):
        return 0, len(self._sample_shape)

    @staticmethod
    def sample_dims_(dict_):
        SampleDict._check_dict_type(dict_)
        return dict_.sample_dims

    @property
    def sample_shape(self):
        return self._sample_shape

    @staticmethod
    def sample_shape_(dict_):
        SampleDict._check_dict_type(dict_)
        return dict_.sample_shape

    def features_dims(self, var):
        if isinstance(var, str):
            var = self[var]
        return len(self._sample_shape), var.ndim

    @staticmethod
    def features_dims_(dict_, var_name):
        SampleDict._check_dict_type(dict_)
        return dict_.features_dims(var_name)

    def features_shape(self, var):
        if isinstance(var, str):
            var_name = var
            shape = self[var_name].shape
        else:
            shape = var.shape
        return shape[slice(*self.features_dims(var))]

    @staticmethod
    def features_shape_(dict_, var_name):
        SampleDict._check_dict_type(dict_)
        return dict_.features_shape_(var_name)

    @property
    def batch_dim(self):
        return len(self._sample_shape) - 1

    @staticmethod
    def batch_dim_(dict_):
        SampleDict._check_dict_type(dict_)
        return dict_.batch_dim

    def n_batch(self, var_name):
        return self[var_name].shape[self.batch_dim]

    @staticmethod
    def n_batch_(dict_, var_name):
        SampleDict._check_dict_type(dict_)
        return dict_.n_batch_(var_name)

    # def squeeze(self, dim=(0,)):
    #     pass

    # def sample_shape_unsqueezed(self):
    #     pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__() + f" --(sample_shape={repr(list(self._sample_shape))})"

    def __eq__(self, other):
        self._check_dict_type(other)
        return super() == other and self.sample_shape == other.sample_shape

    def __ne__(self, other):
        return not self == other

    def update(self, variables, **kwargs) -> None:
        if isinstance(variables, SampleDict):
            target_sample_shape = variables.sample_shape
        else:
            target_sample_shape = self.sample_shape
        ndim_target_s = len(target_sample_shape)
        n_unsqueeze = len(self.sample_shape) - ndim_target_s

        if n_unsqueeze < 0:
            for key in self:
                super().__setitem__(key, self[key][(None,) * -n_unsqueeze])
            if self.sample_shape:
                self._sample_shape = target_sample_shape[:-len(self.sample_shape)] + self.sample_shape
            else:
                self._sample_shape = target_sample_shape

        if isinstance(variables, dict):
            variables = variables.items()
        for key, value in chain(variables, kwargs.items()):
            tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float)
            assert self.is_broadcastable_to(tensor.shape[:ndim_target_s], self.sample_shape[-ndim_target_s:]), \
                f"the sample shape of a new item does not match. item: {value}," \
                f" expected shape: {self.sample_shape[-ndim_target_s:]}, actual shape: {tensor.shape[:ndim_target_s]}"
            if n_unsqueeze >= 0:
                super().__setitem__(key, tensor[(None,) * n_unsqueeze])
            else:
                super().__setitem__(key, tensor)

    def copy(self):
        return SampleDict(self)

    @staticmethod
    def from_arg(arg, required_keys=()):
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
        if isinstance(arg, SampleDict):
            if not (set(list(arg.keys())) >= set(required_keys)):
                raise ValueError("Input keys are not valid.")
            return SampleDict(arg)

        if arg is None:
            if len(required_keys) != 0:
                raise ValueError("arg has no required keys.")
            return SampleDict()

        value_list = None
        if torch.is_tensor(arg) or isinstance(arg, Number):
            value_list = [arg]
        elif isinstance(arg, Iterable) and not isinstance(arg, dict):
            value_list = arg

        if value_list:
            if len(required_keys) != len(value_list):
                raise ValueError("the number of items must be the same as that of keys")
            items = zip(required_keys, value_list)
        elif isinstance(arg, dict):
            if not (set(list(arg.keys())) >= set(required_keys)):
                raise ValueError("Input keys are not valid.")
            items = arg.items()
        else:
            raise ValueError(f"The type of input is not valid, got {type(arg)}.")
        result = [(key, value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float))
                  for key, value in items]
        sample_shape = torch.Size()
        if len(result) > 0 and result[0][1].dim() >= 2:
            sample_shape = (result[0][1].shape[0],)
        result = SampleDict(result, sample_shape=sample_shape)
        # result = SampleDict({key: value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float)
        #                      for key, value in items})

        return result

    def fromkeys(self, seq):
        raise NotImplementedError("fromkeys is not supported on SampleDict. please use from_variables instead of it."
                                  "(because it does not assign default value to unknown keys.)")

    def from_variables(self, var):
        """Get values from `dicts` specified by `keys`.

        When `return_dict` is True, return values are in dictionary format.

        Parameters
        ----------
        var : Iterable[str]

        Returns
        -------
        SampleDict[str, Tensor]

        Examples
        --------
        >>> SampleDict({"a":1,"b":2,"c":3}).from_variables(["b"])
        {'b': tensor(2.)} --(sample_shape=[])
        """
        return SampleDict(((var_name, sample) for var_name, sample in self.items() if var_name in var),
                          self.sample_shape)

    @staticmethod
    def from_variables_(dict_: 'SampleDict', var: List[str]):
        dict_ = SampleDict.from_arg(dict_)
        return dict_.from_variables(var)

    def replaced_dict(self, key_diff: Dict[str, str]):
        return SampleDict({key if key not in key_diff else key_diff[key]: value for key, value in self.items()},
                          self.sample_shape)

    @staticmethod
    def replaced_dict_(dict_: 'SampleDict', key_diff: Dict[str, str]):
        dict_ = SampleDict.from_arg(dict_)
        return dict_.replaced_dict(key_diff)

    def split(self, var):
        extracted_sample = SampleDict(((var_name, sample)
                                      for var_name, sample in self.items() if var_name in var), self.sample_shape)

        remain_sample = SampleDict(((var_name, sample)
                                   for var_name, sample in self.items() if var_name not in var), self.sample_shape)
        return extracted_sample, remain_sample

    @staticmethod
    def split_(dict_, keys):
        """ Splitted dictionaries by :attr:`keys`.

        Parameters
        ----------
        dict_ : dict or SampleDict or None
            Dictionary.

        Returns
        -------
        dict_with_keys : SampleDict
            Dictionary.
        remain_dict : SampleDict
            Dictionary.

        Examples
        --------
        >>> replace_list_dict = {'a': 'loc'}
        >>> x_dict = SampleDict({'a': 0, 'b': 1})
        >>> replacing, remain = SampleDict.split_(x_dict, replace_list_dict.keys())
        >>> replaced = SampleDict.replaced_dict_(replacing, replace_list_dict)
        >>> print((replaced, remain))
        ({'loc': tensor(0.)} --(sample_shape=[]), {'b': tensor(1.)} --(sample_shape=[]))
        """
        dict_ = SampleDict.from_arg(dict_)
        return dict_.split(keys)

    @staticmethod
    def list_from_variables(dict_, vars):
        dict_ = SampleDict.from_arg(dict_)
        return list(dict_[key] for key in vars if key in dict_)

    def detach(self):
        """Return new SampleDict whose values are detached.

        Returns
        -------
        SampleDict
        """
        return SampleDict(((var_name, sample.detach()) for var_name, sample in self.items()), self.sample_shape)
