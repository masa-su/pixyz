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

    def _get_estimated_value(self, x, **kwargs):
        if self._p1.distribution_name == "Normal" and self._p2.distribution_name == "Normal":
            inputs = get_dict_values(x, self._p1.input_var, True)
            params1 = self._p1.get_params(inputs, **kwargs)

            inputs = get_dict_values(x, self._p2.input_var, True)
            params2 = self._p2.get_params(inputs, **kwargs)

            return gauss_gauss_kl(params1["loc"], params1["scale"],
                                  params2["loc"], params2["scale"]), x

        if (self._p1.distribution_name == "vonMisesFisher" and \
            self._p2.distribution_name == "HypersphericalUniform"):
            inputs = get_dict_values(x, self._p1.input_var, True)
            params1 = self._p1.get_params(inputs, **kwargs)
            
            hyu_dim = self._p2.dim
            return vmf_hyu_kl(params1["loc"], params1["scale"],
                              hyu_dim, self.device), x
        
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

def vmf_hyu_kl(vmf_loc, vmf_scale, hyu_dim, device):
    __m = vmf_loc.shape[-1]
    vmf_entropy = vmf_scale * ive(__m/2, vmf_scale) / ive((__m/2)-1, vmf_scale)
    vmf_log_norm = ((__m / 2 - 1) * torch.log(vmf_scale) - (__m / 2) * math.log(2 * math.pi) - (
        vmf_scale + torch.log(ive(__m / 2 - 1, vmf_scale))))
    vmf_log_norm = vmf_log_norm.view(*(vmf_log_norm.shape[:-1]))
    vmf_entropy = vmf_entropy.view(*(vmf_entropy.shape[:-1])) + vmf_log_norm

    hyu_entropy = math.log(2) + ((hyu_dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
        torch.Tensor([(hyu_dim + 1) / 2])).to(device)
    return - vmf_entropy + hyu_entropy
