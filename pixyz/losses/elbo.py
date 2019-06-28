import sympy
import torch
from .losses import SetLoss, Loss


class ELBO(SetLoss):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64]) # q(z|x)
    >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"], features_shape=[64]) # p(x|z)
    >>> loss_cls = ELBO(p, q)
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) - \log p(z|x) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    """
    def __init__(self, p, q, input_var=None):

        loss = (p.log_prob() - q.log_prob()).expectation(q, input_var)
        super().__init__(loss)


class IWELBO(Loss):
    r"""
    The lower bound of importance weighted auto-encoders (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{z_1,z_2,...,z_K \sim q(z|x)}[\log \frac{1}{K} \sum_{k=1}{K} \frac{p(x,z_k)}{q(z_k|x)}]

    where :math:`z_k \sim q(z|x)`.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64]) # q(z|x)
    >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"], features_shape=[64]) # p(x|z)
    >>> loss_cls = IWELBO(p, q)
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) - \log p(z|x) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    """
    def __init__(self, p, q, input_var=None, iw_sample_shape=torch.Size(), sample_shape=torch.Size()):
        self.w = p.log_prob() - q.log_prob()

        if input_var is None:
            input_var = list(set(p.input_var) | set(self.w.input_var) - set(p.var))
        self.sample_shape = torch.Size(sample_shape)
        self.iw_sample_shape = torch.Size(iw_sample_shape)

        super().__init__(p, q, input_var=input_var)

    @property
    def _symbol(self):
        q_text = "{" + self.q.prob_text + "}"
        w_text = "\\frac {} {}".format(
            "{" + self.p.prob_text + "}",
            "{" + self.q.prob_text + "}")

        if self.iw_sample_shape.numel() == 1:
            logsum_w_text = self.w.loss_text

        else:
            logsum_w_text = "\\log \\left(\\frac{{1}}{} \\sum_{{k=1}}^{} \\frac {} {} \\right)".format(
                "{" + str(self.iw_sample_shape.numel()) + "}",
                "{" + str(self.iw_sample_shape.numel()) + "}",
                "{" + self.p.prob_text + "}",
                "{" + self.q.prob_text + "}")

        return sympy.Symbol("\\mathbb{{E}}_{} \\left[{} \\right]".format(q_text, logsum_w_text))

    def _get_eval(self, x_dict={}, **kwargs):
        samples_dict = self.q.sample(x_dict, sample_shape=self.iw_sample_shape + self.sample_shape,
                                     reparam=True, return_all=True)

        loss, loss_sample_dict = self.w.eval(samples_dict, return_dict=True, **kwargs)  # TODO: eval or _get_eval
        samples_dict.update(loss_sample_dict)

        # mean over iw_sample_shape
        loss = loss.view(self.iw_sample_shape.numel(), -1)
        loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(self.iw_sample_shape.numel(), dtype=torch.float32))

        # sum over sample_shape
        loss = loss.view(self.sample_shape.numel(), -1).mean(dim=0)

        return loss, samples_dict
