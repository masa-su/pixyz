from ..utils import get_dict_values
from .distributions import Distribution


class CustomPDF(Distribution):
    """This distribution is constructed by user-defined probability density function.

    Note that this distribution cannot perform sampling.

    Attributes
    ----------
    pdf : function
        User-defined probability density function.
    var : list
        Variables of this distribution.
    distribution_name : :obj:`str`, optional
        Name of this distribution.
    +*kwargs :
        Arbitrary keyword arguments.

    """

    def __init__(self, pdf, var, distribution_name="Custom PDF",
                 **kwargs):
        self.pdf = pdf
        self.DistributionTorch = None
        self._distribution_name = distribution_name

        super().__init__(var=var, cond_var=[], **kwargs)

    @property
    def input_var(self):
        """
        In CustomPDF, :attr:`input_var` is same as :attr:`var`.
        """

        return self.var

    @property
    def distribution_name(self):
        return self._distribution_name

    def get_log_prob(self, x_dict, sum_features=False, feature_dims=None):
        x = get_dict_values(x_dict, self._var)
        return self.pdf(x)
