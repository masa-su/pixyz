import torch

from ..utils import get_dict_values
# TODO: it uses old API


class KullbackLeibler(object):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

        self.q1_name = self.q1.distribution_name
        self.q2_name = self.q2.distribution_name

    def estimate(self, x):
        if self.q1_name == "Normal" and self.q2_name == "Normal":
            inputs = get_dict_values(x, self.q1.cond_var)
            if len(inputs) > 0:
                loc1, scale1 = self.q1.forward(*inputs)
            else:
                loc1 = self.q1.loc
                scale1 = self.q1.scale

            inputs = get_dict_values(x, self.q2.cond_var)
            if len(inputs) > 0:
                loc2, scale2 = self.q2.forward(*inputs)
            else:
                loc2 = self.q2.loc
                scale2 = self.q2.scale
            return gauss_gauss_kl(loc1, scale1, loc2, scale2)

        raise Exception("You cannot use these distributions, "
                        "got %s and %s." % (self.q1_name,
                                            self.q2_name))


def gauss_gauss_kl(loc1, scale1, loc2, scale2):
    _kl = torch.log(scale2) - torch.log(scale1) \
            + (scale1 + (loc1 - loc2)**2) / scale2 - 1
    return 0.5 * torch.sum(_kl, dim=1)
