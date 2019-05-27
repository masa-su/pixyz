import sympy
import torch
from torch.distributions import kl_divergence

from ..utils import get_dict_values
from .losses import Loss


class KullbackLeibler(Loss):
    r"""
    Kullback-Leibler divergence (analytical).

    .. math::

        D_{KL}[p||q] = \mathbb{E}_{p(x)}[\log \frac{p(x)}{q(x)}]

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal, Beta
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[64], name="p")
    >>> q = Beta(concentration0=torch.tensor(1.), concentration1=torch.tensor(1.),
    ...          var=["z"], features_shape=[64], name="q")
    >>> loss_cls = KullbackLeibler(p, q)
    >>> print(loss_cls)
    D_{KL} \left[p(z)||q(z) \right]
    >>> loss = loss_cls.eval()
    """

    def __init__(self, p, q, input_var=None, dim=None):
        self.dim = dim
        super().__init__(p, q, input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("D_{{KL}} \\left[{}||{} \\right]".format(self.p.prob_text, self.q.prob_text))

    def _get_eval(self, x_dict, **kwargs):
        if (not hasattr(self.p, 'distribution_torch_class')) or (not hasattr(self.q, 'distribution_torch_class')):
            raise ValueError("Divergence between these two distributions cannot be evaluated, "
                             "got %s and %s." % (self.p.distribution_name, self.q.distribution_name))

        input_dict = get_dict_values(x_dict, self.p.input_var, True)
        self.p.set_dist(input_dict)

        input_dict = get_dict_values(x_dict, self.q.input_var, True)
        self.q.set_dist(input_dict)

        divergence = kl_divergence(self.p.dist, self.q.dist)

        if self.dim:
            divergence = torch.sum(divergence, dim=self.dim)
            return divergence, x_dict

        dim_list = list(torch.arange(divergence.dim()))
        divergence = torch.sum(divergence, dim=dim_list[1:])
        return divergence, x_dict

        """
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

        #inputs = get_dict_values(x, self._p2.input_var, True)
        #self._p2.set_dist(inputs)

        #divergence = kl_divergence(self._p1.dist, self._p2.dist)

        if self.dim:
            _kl = torch.sum(divergence, dim=self.dim)
            return divergence, x
        """


"""
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
"""
