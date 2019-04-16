from ..utils import get_dict_values
from .distributions import Distribution


class CustomPDF(Distribution):

    def __init__(self, var=["x"], pdf=None, distribution_name="Custom PDF",
                 **kwargs):
        if pdf is None:
            raise ValueError("You should set a pdf.")
        self.pdf = pdf
        self.DistributionTorch = None
        self._distribution_name = distribution_name

        super().__init__(var=var, cond_var=[], **kwargs)

    @property
    def input_var(self):
        """
        In CustomPDF, `input_var` is same as `var`.
        """

        return self.var

    @property
    def distribution_name(self):
        return self._distribution_name

    def get_log_prob(self, x_dict, sum_features=False, feature_dims=None):
        x = get_dict_values(x_dict, self._var)
        return self.pdf(x)
