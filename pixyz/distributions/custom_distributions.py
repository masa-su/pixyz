from ..utils import get_dict_values
from .distributions import Distribution


class CustomProb(Distribution):
    """This distribution is constructed by user-defined probability density/mass function.

    Note that this distribution cannot perform sampling.

    """

    def __init__(self, log_prob_function, var, distribution_name="Custom PDF", **kwargs):
        """
        Parameters
        ----------
        log_prob_function : function
            User-defined log-probability density/mass function.
        var : list
            Variables of this distribution.
        distribution_name : :obj:`str`, optional
            Name of this distribution.
        +*kwargs :
            Arbitrary keyword arguments.

        """
        self._log_prob_function = log_prob_function
        self._distribution_name = distribution_name

        super().__init__(var=var, cond_var=[], **kwargs)

    @property
    def log_prob_function(self):
        """User-defined log-probability density/mass function."""
        return self._log_prob_function

    @property
    def input_var(self):
        return self.var

    @property
    def distribution_name(self):
        return self._distribution_name

    def get_log_prob(self, x_dict, sum_features=False, feature_dims=None):
        x = get_dict_values(x_dict, self._var, return_dict=True)
        return self.log_prob_function(**x)
