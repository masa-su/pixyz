import torch
from .losses import Loss
from ..utils import get_dict_values


class MMD(Loss):
    r"""
    Maximum Mean Discrepancy (MMD).

    .. math::

        D_{MMD}[p||q] = \mathbb{E}_{p(x), p(x')}[k(x, x')] + \mathbb{E}_{q(x), q(x')}[k(x, x')]
        - 2\mathbb{E}_{p(x), q(x')}[k(x, x')]

    where :math:`k(x, x')` is any positive definite kernel.
    """
    def __init__(self, p, q, input_var=None, kernel="rbf", **kernel_params):
        if p.var != q.var:
            raise ValueError("The two distribution variables must be the same.")

        if len(p.var) != 1:
            raise ValueError("A given distribution must have only one variable.")

        if len(p.input_var) > 0:
            self.input_dist = p
        elif len(q.input_var) > 0:
            self.input_dist = q
        else:
            raise NotImplementedError

        if kernel == "rbf":
            self.kernel = rbf_kernel
        else:
            raise NotImplementedError

        self.kernel_params = kernel_params

        if input_var is None:
            input_var = p.input_var + q.input_var

        super().__init__(p, q, input_var=input_var)

    @property
    def loss_text(self):
        return "MMD[{}||{}]".format(self._p.prob_text, self._q.prob_text)

    def _get_estimated_value(self, x={}, **kwargs):
        batch_size = get_dict_values(x, self.input_dist.input_var[0])[0].shape[0]

        # sample from distributions
        p_x = get_dict_values(self._p.sample(x, batch_size=batch_size), self._p.var)[0]
        q_x = get_dict_values(self._q.sample(x, batch_size=batch_size), self._q.var)[0]

        if p_x.shape != q_x.shape:
            raise ValueError("The two distribution variables must have the same shape.")

        if len(p_x.shape) != 2:
            raise ValueError("The number of axes of a given sample must be 2, got %d" % len(p_x.shape))

        # estimate MMD
        p_kernel = self.kernel(p_x, p_x, **self.kernel_params).mean()
        q_kernel = self.kernel(q_x, q_x, **self.kernel_params).mean()
        pq_kernel = self.kernel(p_x, q_x, **self.kernel_params).mean()
        mmd_loss = p_kernel + q_kernel - 2*pq_kernel

        return mmd_loss, x


def rbf_kernel(x, y, sigma_sqr=1, **kwargs):
    r"""
    Radial basis function (RBF) kernel.

    .. math::

        k(x, x') = \exp (\frac{||x-x'||^2}{2\sigma^2})
    """

    kernel_input = ((x[:, None, :] - y[None, :, :])**2).mean(-1)
    return torch.exp(-kernel_input / 2.0*sigma_sqr)





