from __future__ import print_function

from .distributions import Distribution
from ..utils import get_dict_values


class Deterministic(Distribution):
    """
    Deterministic distribution (or degeneration distribution)

    Examples
    --------
    >>> import torch
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["z"], var=["x"])
    ...         self.model = torch.nn.Linear(64, 512)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p = Generator()
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=512, bias=True)
      )
    >>> sample = p.sample({"z": torch.randn(1, 64)})
    >>> p.log_prob().eval(sample) # log_prob is not defined.
    Traceback (most recent call last):
     ...
    NotImplementedError
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def distribution_name(self):
        return "Deterministic"

    def sample(self, x_dict={}, return_all=True, **kwargs):
        x_dict = self._check_input(x_dict)
        _x_dict = get_dict_values(x_dict, self.input_var, return_dict=True)
        output_dict = self.forward(**_x_dict)

        if set(output_dict.keys()) != set(self._var):
            raise ValueError("Output variables are not same as `var`.")

        if return_all:
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]


class DataDistribution(Distribution):
    """
    Data distribution.

    Samples from this distribution equal given inputs.

    Examples
    --------
    >>> import torch
    >>> p = DataDistribution(var=["x"])
    >>> print(p)
    Distribution:
      p_{data}(x)
    Network architecture:
      DataDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> sample = p.sample({"x": torch.randn(1, 64)})
    """

    def __init__(self, var, name="p_{data}"):
        super().__init__(var=var, cond_var=[], name=name)

    @property
    def distribution_name(self):
        return "Data distribution"

    def sample(self, x_dict={}, **kwargs):
        output_dict = self._check_input(x_dict)
        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]

    @property
    def input_var(self):
        """
        In DataDistribution, `input_var` is same as `var`.
        """

        return self.var
