import torch
from ..utils import get_dict_values
from .losses import Loss


class KullbackLeibler(Loss):
    r"""
    Kullback-Leibler divergence (analytical).

    .. math::

        D_{KL}[p||q] = \mathbb{E}_{p(x)}[\log \frac{p(x)}{q(x)}]
    """

    def __init__(self, p1, p2, input_var=None):
        super().__init__(p1, p2, input_var)

    @property
    def loss_text(self):
        return "KL[{}||{}]".format(self._p1.prob_text, self._p2.prob_text)

    def estimate(self, x, **kwargs):
        x = super().estimate(x)

        if self._p1.distribution_name == "Normal" and self._p2.distribution_name == "Normal":
            inputs = get_dict_values(x, self._p1.input_var, True)
            params1 = self._p1.get_params(inputs, **kwargs)

            inputs = get_dict_values(x, self._p2.input_var, True)
            params2 = self._p2.get_params(inputs, **kwargs)

            return gauss_gauss_kl(params1["loc"], params1["scale"],
                                  params2["loc"], params2["scale"])

        raise Exception("You cannot use these distributions, "
                        "got %s and %s." % (self._p1.distribution_name,
                                            self._p2.distribution_name))


def gauss_gauss_kl(loc1, scale1, loc2, scale2, dim=None):
    # https://github.com/pytorch/pytorch/blob/85408e744fc1746ab939ae824a26fd6821529a94/torch/distributions/kl.py#L384
    var_ratio = (scale1 / scale2).pow(2)
    t1 = ((loc1 - loc2) / scale2).pow(2)
    _kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

    if dim:
        _kl = torch.sum(_kl, dim=dim)
        return _kl

    dim_list = list(torch.arange(_kl.dim()))
    _kl = torch.sum(_kl, dim=dim_list[1:])
    return _kl
