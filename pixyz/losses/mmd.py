import torch
import sympy
from .losses import Loss
from ..utils import get_dict_values


class MMD(Loss):
    r"""
    The Maximum Mean Discrepancy (MMD).

    .. math::

        D_{MMD^2}[p||q] = \mathbb{E}_{p(x), p(x')}[k(x, x')] + \mathbb{E}_{q(x), q(x')}[k(x, x')]
        - 2\mathbb{E}_{p(x), q(x')}[k(x, x')]

    where :math:`k(x, x')` is any positive definite kernel.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64], name="p")
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64], name="q")
    >>> loss_cls = MMD(p, q, kernel="gaussian")
    >>> print(loss_cls)
    D_{MMD^2} \left[p(z|x)||q(z|x) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    >>> # Use the inverse (multi-)quadric kernel
    >>> loss = MMD(p, q, kernel="inv-multiquadratic").eval({"x": torch.randn(10, 64)})
    """

    def __init__(self, p, q, input_var=None, kernel="gaussian", **kernel_params):
        if p.var != q.var:
            raise ValueError("The two distribution variables must be the same.")

        if len(p.var) != 1:
            raise ValueError("A given distribution must have only one variable.")

        if len(p.input_var) > 0:
            self.input_dist = p
        elif len(q.input_var) > 0:
            self.input_dist = q
        else:
            raise NotImplementedError()

        if kernel == "gaussian":
            self.kernel = gaussian_rbf_kernel
        elif kernel == "inv-multiquadratic":
            self.kernel = inverse_multiquadratic_rbf_kernel
        else:
            raise NotImplementedError()

        self.kernel_params = kernel_params

        if input_var is None:
            input_var = p.input_var + q.input_var

        super().__init__(p, q, input_var=input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("D_{{MMD^2}} \\left[{}||{} \\right]".format(self.p.prob_text, self.q.prob_text))

    def _get_batch_n(self, x_dict):
        return get_dict_values(x_dict, self.input_dist.input_var[0])[0].shape[0]

    def _get_eval(self, x_dict={}, **kwargs):
        batch_n = self._get_batch_n(x_dict)

        # sample from distributions
        p_x = get_dict_values(self.p.sample(x_dict, batch_n=batch_n), self.p.var)[0]
        q_x = get_dict_values(self.q.sample(x_dict, batch_n=batch_n), self.q.var)[0]

        if p_x.shape != q_x.shape:
            raise ValueError("The two distribution variables must have the same shape.")

        if len(p_x.shape) != 2:
            raise ValueError("The number of axes of a given sample must be 2, got %d" % len(p_x.shape))

        p_x_dim = p_x.shape[1]
        q_x_dim = q_x.shape[1]

        # estimate the squared MMD (unbiased estimator)
        p_kernel = self.kernel(p_x, p_x, **self.kernel_params).sum() / (p_x_dim * (p_x_dim - 1))
        q_kernel = self.kernel(q_x, q_x, **self.kernel_params).sum() / (q_x_dim * (q_x_dim - 1))
        pq_kernel = self.kernel(p_x, q_x, **self.kernel_params).sum() / (p_x_dim * q_x_dim)
        mmd_loss = p_kernel + q_kernel - 2 * pq_kernel

        return mmd_loss, x_dict


def pairwise_distance_matrix(x, y, metric="euclidean"):
    r"""
    Computes the pairwise distance matrix between x and y.
    """

    if metric == "euclidean":
        return torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1)

    raise NotImplementedError()


def gaussian_rbf_kernel(x, y, sigma_sqr=2., **kwargs):
    r"""
    Gaussian radial basis function (RBF) kernel.

    .. math::

        k(x, y) = \exp (\frac{||x-y||^2}{\sigma^2})
    """

    return torch.exp(-pairwise_distance_matrix(x, y) / (1. * sigma_sqr))


def inverse_multiquadratic_rbf_kernel(x, y, sigma_sqr=2., **kwargs):
    r"""
    Inverse multi-quadratic radial basis function (RBF) kernel.

    .. math::

        k(x, y) = \frac{\sigma^2}{||x-y||^2 + \sigma^2}
    """

    return sigma_sqr / (pairwise_distance_matrix(x, y) + sigma_sqr)
