from .losses import Loss, SetLoss


class Expectation(Loss):
    r"""
    Expectation of a given function (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^L f(x_l),

    where :math:`x_l \sim p(x)`.

    Note that :math:`f` doesn't need to be able to sample, which is known as the law of the unconscious statistician
     (LOTUS).

    Therefore, in this class, :math:`f` is assumed to `pixyz.Loss`.
    """

    def __init__(self, p, f, input_var=None):

        if input_var is None:
            input_var = list(set(p.input_var) | set(f.input_var) - set(p.var))
        self._f = f

        super().__init__(p, input_var=input_var)

    @property
    def loss_text(self):
        return "E_{}[{}]".format(self._p.prob_text, self._f.loss_text)

    def _get_estimated_value(self, x={}, **kwargs):
        samples_dict = self._p.sample(x, reparam=True, return_all=True)

        # TODO: whether estimate or _get_estimate_value
        loss, loss_sample_dict = self._f.estimate(samples_dict, return_dict=True, **kwargs)
        samples_dict.update(loss_sample_dict)

        return loss, samples_dict


class CrossEntropy(SetLoss):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation).

    .. math::

        H[p||q] = -\mathbb{E}_{p(x)}[\log q(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log q(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the `Expectation` class.
    """

    def __init__(self, p, q, input_var=None):
        if input_var is None:
            input_var = list(set(p.input_var + q.var))

        loss = -Expectation(p, q.log_prob(), input_var)
        super().__init__(loss)


class Entropy(SetLoss):
    r"""
    Entropy (Monte Carlo approximation).

    .. math::

        H[p] = -\mathbb{E}_{p(x)}[\log p(x)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l),

    where :math:`x_l \sim p(x)`.

    Note:
        This class is a special case of the `Expectation` class.
    """

    def __init__(self, p, input_var=None):
        if input_var is None:
            input_var = p.input_var

        loss = -Expectation(p, p.log_prob(), input_var)
        super().__init__(loss)


class StochasticReconstructionLoss(SetLoss):
    r"""
    Reconstruction Loss (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x|z_l),

    where :math:`z_l \sim q(z|x)`.

    Note:
        This class is a special case of the `Expectation` class.
    """

    def __init__(self, encoder, decoder, input_var=None):

        if input_var is None:
            input_var = encoder.input_var

        if not(set(decoder.var) <= set(input_var)):
            raise ValueError("Variable {} (in the `{}` class) is not included"
                             " in `input_var` of the `{}` class.".format(decoder.var,
                                                                         decoder.__class__.__name__,
                                                                         encoder.__class__.__name__))

        loss = -Expectation(encoder, decoder.log_prob(), input_var)
        super().__init__(loss)
