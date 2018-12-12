from .losses import Loss
from ..utils import get_dict_values


class CrossEntropy(Loss):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(x)}[\log p(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l),

    where :math:`x_l \sim q(x)`.
    """

    def __init__(self, p1, p2, input_var=None):
        if input_var is None:
            input_var = list(set(p1.input_var + p2.var))
        super().__init__(p1, p2, input_var=input_var)

    @property
    def loss_text(self):
        return "-E_{}[log {}]".format(self._p1.prob_text, self._p2.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        _p1_input = get_dict_values(_x, self._p1.input_var, return_dict=True)
        samples = self._p1.sample(_p1_input, reparam=True, return_all=False)

        _p2_input = get_dict_values(_x, self._p2.var, return_dict=True)
        samples.update(_p2_input)

        loss = -self._p2.log_likelihood(samples)

        return loss


class Entropy(Loss):
    r"""
    Entropy (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{p(x)}[\log p(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the `CrossEntropy` class. You can get the same result with `CrossEntropy`.
    """

    def __init__(self, p1, input_var=None):
        if input_var is None:
            input_var = p1.input_var
        super().__init__(p1, None, input_var=input_var)

    @property
    def loss_text(self):
        return "-E_{}[log {}]".format(self._p1.prob_text, self._p1.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        samples = self._p1.sample(_x, reparam=True)

        loss = self._p1.log_likelihood(samples)

        return loss


class StochasticReconstructionLoss(Loss):
    r"""
    Reconstruction Loss (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x|z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the `CrossEntropy` class. You can get the same result with `CrossEntropy`.
    """

    def __init__(self, encoder, decoder, input_var=None):

        if input_var is None:
            input_var = encoder.input_var
        super().__init__(encoder, decoder, input_var=input_var)

    @property
    def loss_text(self):
        return "-E_{}[log {}]".format(self._p1.prob_text, self._p2.prob_text)

    def estimate(self, x={}):
        _x = super().estimate(x)
        samples = self._p1.sample(_x, reparam=True)
        loss = -self._p2.log_likelihood(samples)

        return loss
