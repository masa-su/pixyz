from ..utils import get_dict_values, sum_samples
from .distributions import Distribution


class CustomProb(Distribution):
    """This distribution is constructed by user-defined probability density/mass function.

    Note that this distribution cannot perform sampling.

    Examples
    --------
    >>> import torch
    >>> # banana shaped distribution
    >>> def log_prob(z):
    ...     z1, z2 = torch.chunk(z, chunks=2, dim=1)
    ...     norm = torch.sqrt(z1 ** 2 + z2 ** 2)
    ...     exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    ...     exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    ...     u = 0.5 * ((norm - 2) / 0.4) ** 2 - torch.log(exp1 + exp2)
    ...     return -u
    ...
    >>> p = CustomProb(log_prob, var=["z"])
    >>> loss = p.log_prob().eval({"z": torch.randn(10, 2)})
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

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        x_dict = get_dict_values(x_dict, self._var, return_dict=True)
        log_prob = self.log_prob_function(**x_dict)
        if sum_features:
            log_prob = sum_samples(log_prob)

        return log_prob
