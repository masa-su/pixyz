from __future__ import print_function

from .distributions import Distribution
from ..utils import get_dict_values


class Deterministic(Distribution):
    """
    Deterministic distribution (or degeneration distribution)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def distribution_name(self):
        return "Deterministic"

    def sample(self, x={}, return_all=True, **kwargs):
        x_dict = self._check_input(x)
        _x_dict = get_dict_values(x_dict, self.input_var, return_dict=True)
        output_dict = self.forward(**_x_dict)

        if set(output_dict.keys()) != set(self._var):
            raise ValueError("Output variables are not same as `var`.")

        if return_all:
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def sample_mean(self, x):
        return self.sample(x, return_all=False)[self._var[0]]


class DataDistribution(Distribution):
    """
    Data distribution.
    TODO: Fix this behavior if multiplied with other distributions
    """

    def __init__(self, var, name="p_{data}"):
        super().__init__(var=var, cond_var=[], name=name, dim=1)

    @property
    def distribution_name(self):
        return "Data distribution"

    def sample(self, x={}, **kwargs):
        output_dict = self._check_input(x)
        return output_dict

    def sample_mean(self, x):
        return self.sample(x, return_all=False)[self._var[0]]

    @property
    def input_var(self):
        """
        In DataDistribution, `input_var` is same as `var`.
        """

        return self.var
