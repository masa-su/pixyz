import torch
from ..utils import get_dict_values
from .losses import Loss


class KullbackLeibler(Loss):
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


def gauss_gauss_kl(loc1, scale1, loc2, scale2):
    _kl = torch.log(scale2) - torch.log(scale1) \
            + (scale1**2 + (loc1 - loc2)**2) / scale2**2 - 1/2
    for _ in range(1, loc1.dim()):
        _kl = torch.sum(_kl, dim=-1)
    return _kl
