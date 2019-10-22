from torch.nn.modules.distance import PairwiseDistance
import sympy
from .losses import Loss


class WassersteinDistance(Loss):
    r"""
    Wasserstein distance.

    .. math::

         W(p, q) = \inf_{\Gamma \in \mathcal{P}(x_p\sim p, x_q\sim q)} \mathbb{E}_{(x_p, x_q) \sim \Gamma}[d(x_p, x_q)]

    However, instead of the above true distance, this class computes the following one.

    .. math::

         W'(p, q) = \mathbb{E}_{x_p\sim p, x_q \sim q}[d(x_p, x_q)].


    Here, :math:`W'` is the upper of :math:`W` (i.e., :math:`W\leq W'`), and these are equal when both :math:`p`
    and :math:`q` are degenerate (deterministic) distributions.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> p = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64], name="p")
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"], features_shape=[64], name="q")
    >>> loss_cls = WassersteinDistance(p, q)
    >>> print(loss_cls)
    W^{upper} \left(p(z|x), q(z|x) \right)
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    """

    def __init__(self, p, q, metric=PairwiseDistance(p=2), input_var=None):
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

        self.metric = metric

        if input_var is None:
            input_var = p.input_var + q.input_var

        super().__init__(p, q, input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("W^{{upper}} \\left({}, {} \\right)".format(self.p.prob_text, self.q.prob_text))

    def _get_eval(self, x_dict, **kwargs):
        # sample from distributions
        p_x = self.p.sample(x_dict)[self.p.var[0]]
        q_x = self.q.sample(x_dict)[self.q.var[0]]

        if p_x.shape != q_x.shape:
            raise ValueError("The two distribution variables must have the same shape.")

        # metric function assumes tensor's shape=(N,D)
        sample_shape = x_dict.sample_shape
        features_shape = x_dict.features_shape(p_x)
        distance = self.metric(p_x.reshape(-1, *features_shape), q_x.reshape(-1, *features_shape))
        if sample_shape:
            distance = distance.reshape(*sample_shape)
        else:
            distance = distance.squeeze(0)

        return distance, x_dict
