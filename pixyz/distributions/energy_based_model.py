import torch
from ..distributions import Distribution
from ..utils import detach_dict


class EnergybasedDistribution(Distribution):
    r"""
     Distribution defined by an energy function :math:`E_{\theta}(\mathbf{x})`:

     .. math::

         p_{\theta}(\mathbf{x})=\frac{\exp \left(-E_{\theta}(\mathbf{x})\right)}{Z(\theta)},

     where :math:`Z(\theta)=\int \exp \left(-E_{\theta}(\mathbf{x})\right) d \mathbf{x}`.


     Once initializing, it can be handled as a distribution module.

     """


    @property
    def distribution_name(self):
        return "Energy-based Model"

    @property
    def has_reparam(self):
        return False

    def langevin(self, x_dict, lam=5e-3):
        _x_dict = dict((key, value.requires_grad_(True)) for key, value in x_dict.items())
        energy = self.get_energy(_x_dict)
        energy.backward()

        return dict((key, value - lam/2. * value.grad + torch.normal(torch.zeros_like(value),
                                                                     torch.ones_like(value) * lam))
                    for key, value in _x_dict.items())

    def get_energy(self, x_dict={}):
        x_dict = self._check_input(x_dict)
        return self.forward(**x_dict)

    def get_partition(self, x_dict={}, mc_iter=10, lam=5e-3):
        sample_dict = detach_dict(self.sample(x_dict, mc_iter=mc_iter, lam=lam))
        return self.get_energy(sample_dict)

    def get_log_prob(self, x_dict={}, mc_iter=10, lam=5e-3):
        energy = self.get_energy(x_dict)
        partition = self.get_partition(x_dict, mc_iter=mc_iter, lam=lam)
        return -energy + torch.log(partition)

    def sample(self, x_dict={}, mc_iter=10, lam=5e-3):
        if len(x_dict) > 0:
            x_dict = self._check_input(x_dict)

        output_dict = x_dict
        for _ in range(mc_iter):
            output_dict = self.langevin(output_dict, lam)

        return output_dict

    def sample_mean(self, x_dict={}):
        return self.sample(x_dict)

    def forward(self):
        raise NotImplementedError()
