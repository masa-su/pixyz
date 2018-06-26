import torch

from ..utils import get_dict_values


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
                mu1, sigma1 = self.q1.forward(*inputs)
            else:
                mu1 = self.q1.mu
                sigma1 = self.q1.sigma

            inputs = get_dict_values(x, self.q2.cond_var)
            if len(inputs) > 0:
                mu2, sigma2 = self.q2.forward(*inputs)
            else:
                mu2 = self.q2.mu
                sigma2 = self.q2.sigma
            return gauss_gauss_kl(mu1, sigma1, mu2, sigma2)

        raise Exception("You cannot use these distributions, "
                        "got %s and %s." % (self.q1_name,
                                            self.q2_name))


def gauss_gauss_kl(mu1, sigma1, mu2, sigma2):
    _kl = torch.log(sigma2) - torch.log(sigma1) \
            + (sigma1 + (mu1 - mu2)**2) / sigma2 - 1
    return 0.5 * torch.sum(_kl, dim=1)
