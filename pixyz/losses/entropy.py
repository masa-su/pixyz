import torch

from ..utils import get_dict_values
from .losses import Loss, SetLoss


class Entropy(SetLoss):
    r"""
    Entropy (Monte Carlo approximation).

    .. math::

        H[p] = -\mathbb{E}_{p(x)}[\log p(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.
    """

    def __init__(self, p, input_var=None):
        if input_var is None:
            input_var = p.input_var

        loss = -p.log_prob().expectation(p, input_var)
        super().__init__(loss)


class AnalyticalEntropy(Loss):
    r"""
    Entropy (Analytical).

    .. math::

        H[p] = -\mathbb{E}_{p(x)}[\log p(x)]
    """

    def __init__(self, p, input_var=None, dim=None):
        self.dim = dim
        super().__init__(p, input_var)

    @property
    def loss_text(self):
        return "KL[{}||{}]".format(self._p.prob_text, self._q.prob_text)

    def _get_eval(self, x, **kwargs):
        if not hasattr(self._p, 'distribution_torch_class'):
            raise ValueError("Entropy of this distribution cannot be evaluated, "
                             "got %s." % self._p.distribution_name)

        inputs = get_dict_values(x, self._p.input_var, True)
        self._p.set_dist(inputs)

        entropy = self._p.dist.entropy()

        if self.dim:
            entropy = torch.sum(entropy, dim=self.dim)
            return entropy, x

        dim_list = list(torch.arange(entropy.dim()))
        entropy = torch.sum(entropy, dim=dim_list[1:])
        return entropy, x


class CrossEntropy(SetLoss):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation).

    .. math::

        H[p||q] = -\mathbb{E}_{p(x)}[\log q(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log q(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.
    """

    def __init__(self, p, q, input_var=None):
        if input_var is None:
            input_var = list(set(p.input_var + q.var))

        loss = -q.log_prob().expectation(p, input_var)
        super().__init__(loss)


class StochasticReconstructionLoss(SetLoss):
    r"""
    Reconstruction Loss (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x|z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the :attr:`Expectation` class.
    """

    def __init__(self, encoder, decoder, input_var=None):

        if input_var is None:
            input_var = encoder.input_var

        if not(set(decoder.var) <= set(input_var)):
            raise ValueError("Variable {} (in the `{}` class) is not included"
                             " in `input_var` of the `{}` class.".format(decoder.var,
                                                                         decoder.__class__.__name__,
                                                                         encoder.__class__.__name__))

        loss = -decoder.log_prob().expectation(encoder, input_var)
        super().__init__(loss)
