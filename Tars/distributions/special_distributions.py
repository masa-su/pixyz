from __future__ import print_function

from .distributions import Distribution


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
        if len(x) > 0:
            x_dict = self._check_input(x)
            output_dict = self.forward(**x_dict)

            if set(output_dict.keys()) != set(self._var):
                raise ValueError("Output variables are not same as `var`.")

            if return_all:
                output_dict.update(x_dict)

            return output_dict

        raise ValueError("You should set inputs.")


class DataDistribution(Distribution):
    """
    Data distribution.
    TODO: Fix this behavior if multiplied with other distributions
    """

    def __init__(self, var, name="p_data"):
        super().__init__(var=var, cond_var=[], name=name, dim=1)

    @property
    def distribution_name(self):
        return "Data distribution"

    def sample(self, x={}, **kwargs):
        if len(x) > 0:
            output_dict = self._check_input(x)
            return output_dict

        raise ValueError("You should set inputs.")

    @property
    def input_var(self):
        """
        In DataDistribution, `input_var` is same as `var`.
        """

        return self.var

