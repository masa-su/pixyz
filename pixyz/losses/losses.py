import abc
import sympy
import torch

import numbers
from copy import deepcopy

from ..utils import tolist


class Loss(object, metaclass=abc.ABCMeta):
    """Loss class. In Pixyz, all loss classes are required to inherit this class.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Bernoulli, Normal
    >>> from pixyz.losses import StochasticReconstructionLoss, KullbackLeibler
    ...
    >>> # Set distributions
    >>> class Inference(Normal):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["x"], var=["z"], name="q")
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    ...
    >>> class Generator(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = torch.nn.Linear(64, 128)
    ...     def forward(self, z):
    ...         return {"probs": torch.sigmoid(self.model(z))}
    ...
    >>> p = Generator()
    >>> q = Inference()
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[64], name="p_{prior}")
    ...
    >>> # Define a loss function (VAE)
    >>> reconst = StochasticReconstructionLoss(q, p)
    >>> kl = KullbackLeibler(q, prior)
    >>> loss_cls = (reconst - kl).mean()
    >>> print(loss_cls)
    mean \\left(- D_{KL} \\left[q(z|x)||p_{prior}(z) \\right] - \\mathbb{E}_{q(z|x)} \\left[\\log p(x|z) \\right] \\right)
    >>> # Evaluate this loss function
    >>> data = torch.randn(1, 128)  # Pseudo data
    >>> loss = loss_cls.eval({"x": data})
    >>> print(loss)  # doctest: +SKIP
    tensor(65.5939, grad_fn=<MeanBackward0>)

    """

    def __init__(self, p, q=None, input_var=None):
        """
        Parameters
        ----------
        p : pixyz.distributions.Distribution
            Distribution.
        q : pixyz.distributions.Distribution, defaults to None
            Distribution.
        input_var : :obj:`list` of :obj:`str`, defaults to None
            Input variables of this loss function.
            In general, users do not need to set them explicitly
            because these depend on the given distributions and each loss function.

        """
        self.p = p
        self.q = q

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = deepcopy(p.input_var)
            if q is not None:
                _input_var += deepcopy(q.input_var)
                _input_var = sorted(set(_input_var), key=_input_var.index)
            self._input_var = _input_var

    @property
    def input_var(self):
        """list: Input variables of this distribution."""
        return self._input_var

    @property
    @abc.abstractmethod
    def _symbol(self):
        raise NotImplementedError()

    @property
    def loss_text(self):
        return sympy.latex(self._symbol)

    def __str__(self):
        return self.loss_text

    def __repr__(self):
        return self.loss_text

    def __add__(self, other):
        return AddLoss(self, other)

    def __radd__(self, other):
        return AddLoss(other, self)

    def __sub__(self, other):
        return SubLoss(self, other)

    def __rsub__(self, other):
        return SubLoss(other, self)

    def __mul__(self, other):
        return MulLoss(self, other)

    def __rmul__(self, other):
        return MulLoss(other, self)

    def __truediv__(self, other):
        return DivLoss(self, other)

    def __rtruediv__(self, other):
        return DivLoss(other, self)

    def __neg__(self):
        return NegLoss(self)

    def abs(self):
        """Return an instance of :class:`pixyz.losses.losses.AbsLoss`.

        Returns
        -------
        pixyz.losses.losses.AbsLoss
            An instance of :class:`pixyz.losses.losses.AbsLoss`

        """
        return AbsLoss(self)

    def mean(self):
        """Return an instance of :class:`pixyz.losses.losses.BatchMean`.

        Returns
        -------
        pixyz.losses.losses.BatchMean
            An instance of :class:`pixyz.losses.BatchMean`

        """
        return BatchMean(self)

    def sum(self):
        """Return an instance of :class:`pixyz.losses.losses.BatchSum`.

        Returns
        -------
        pixyz.losses.losses.BatchSum
            An instance of :class:`pixyz.losses.losses.BatchSum`

        """
        return BatchSum(self)

    def expectation(self, p, input_var=None, sample_shape=torch.Size()):
        """Return an instance of :class:`pixyz.losses.Expectation`.

        Parameters
        ----------
        p : pixyz.distributions.Distribution
            Distribution for sampling.

        input_var : list
            Input variables of this loss.

        sample_shape : :obj:`list` or :obj:`NoneType`, defaults to torch.Size()
            Shape of generating samples.

        Returns
        -------
        pixyz.losses.Expectation
            An instance of :class:`pixyz.losses.Expectation`

        """
        return Expectation(p, self, input_var=input_var, sample_shape=sample_shape)

    def eval(self, x_dict={}, return_dict=False, **kwargs):
        """Evaluate the value of the loss function given inputs (:attr:`x_dict`).

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}
            Input variables.
        return_dict : bool, default to False.
            Whether to return samples along with the evaluated value of the loss function.

        Returns
        -------
        loss : torch.Tensor
            the evaluated value of the loss function.
        x_dict : :obj:`dict`
            All samples generated when evaluating the loss function.
            If :attr:`return_dict` is False, it is not returned.

        """

        if not(set(list(x_dict.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, expected {} but got {}.".format(self._input_var,
                                                                                        list(x_dict.keys())))

        loss, x_dict = self._get_eval(x_dict, **kwargs)

        if return_dict:
            return loss, x_dict

        return loss

    @abc.abstractmethod
    def _get_eval(self, x_dict, **kwargs):
        raise NotImplementedError()


class ValueLoss(Loss):
    """
    This class contains a scalar as a loss value.

    If multiplying a scalar by an arbitrary loss class, this scalar is converted to the :class:`ValueLoss`.


    Examples
    --------
    >>> loss_cls = ValueLoss(2)
    >>> print(loss_cls)
    2
    >>> loss = loss_cls.eval()
    >>> print(loss)
    2

    """
    def __init__(self, loss1):
        self.loss1 = loss1
        self._input_var = []

    def _get_eval(self, x_dict={}, **kwargs):
        return self.loss1, x_dict

    @property
    def _symbol(self):
        return self.loss1


class Parameter(Loss):
    """
    This class defines a single variable as a loss class.

    It can be used such as a coefficient parameter of a loss class.

    Examples
    --------
    >>> loss_cls = Parameter("x")
    >>> print(loss_cls)
    x
    >>> loss = loss_cls.eval({"x": 2})
    >>> print(loss)
    2

    """
    def __init__(self, input_var):
        if not isinstance(input_var, str):
            raise ValueError()
        self._input_var = tolist(input_var)

    def _get_eval(self, x_dict={}, **kwargs):
        return x_dict[self._input_var[0]], x_dict

    @property
    def _symbol(self):
        return sympy.Symbol(self._input_var[0])


class LossOperator(Loss):
    def __init__(self, loss1, loss2):
        _input_var = []

        if isinstance(loss1, Loss):
            _input_var += deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        elif isinstance(loss2, type(None)):
            pass
        else:
            raise ValueError("{} cannot be operated with {}.".format(type(loss1), type(loss2)))

        if isinstance(loss2, Loss):
            _input_var += deepcopy(loss2.input_var)
        elif isinstance(loss2, numbers.Number):
            loss2 = ValueLoss(loss2)
        elif isinstance(loss2, type(None)):
            pass
        else:
            raise ValueError("{} cannot be operated with {}.".format(type(loss2), type(loss1)))

        _input_var = sorted(set(_input_var), key=_input_var.index)

        self._input_var = _input_var
        self.loss1 = loss1
        self.loss2 = loss2

    def _get_eval(self, x_dict={}, **kwargs):
        if not isinstance(self.loss1, type(None)):
            loss1, x1 = self.loss1._get_eval(x_dict, **kwargs)
        else:
            loss1 = 0
            x1 = {}

        if not isinstance(self.loss2, type(None)):
            loss2, x2 = self.loss2._get_eval(x_dict, **kwargs)
        else:
            loss2 = 0
            x2 = {}

        x1.update(x2)

        return loss1, loss2, x1


class AddLoss(LossOperator):
    """
    Apply the `add` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 + loss_cls_2  # equals to AddLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    x + 2
    >>> loss = loss_cls.eval({"x": 3})
    >>> print(loss)
    5

    """
    @property
    def _symbol(self):
        return self.loss1._symbol + self.loss2._symbol

    def _get_eval(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super()._get_eval(x_dict, **kwargs)
        return loss1 + loss2, x_dict


class SubLoss(LossOperator):
    """
    Apply the `sub` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 - loss_cls_2  # equals to SubLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    2 - x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    -2
    >>> loss_cls = loss_cls_2 - loss_cls_1  # equals to SubLoss(loss_cls_2, loss_cls_1)
    >>> print(loss_cls)
    x - 2
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    2

    """
    @property
    def _symbol(self):
        return self.loss1._symbol - self.loss2._symbol

    def _get_eval(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super()._get_eval(x_dict, **kwargs)
        return loss1 - loss2, x_dict


class MulLoss(LossOperator):
    """
    Apply the `mul` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 * loss_cls_2  # equals to MulLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    2 x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    8

    """
    @property
    def _symbol(self):
        return self.loss1._symbol * self.loss2._symbol

    def _get_eval(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super()._get_eval(x_dict, **kwargs)
        return loss1 * loss2, x_dict


class DivLoss(LossOperator):
    """
    Apply the `div` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 / loss_cls_2  # equals to DivLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    \\frac{2}{x}
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    0.5
    >>> loss_cls = loss_cls_2 / loss_cls_1  # equals to DivLoss(loss_cls_2, loss_cls_1)
    >>> print(loss_cls)
    \\frac{x}{2}
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    2.0


    """
    @property
    def _symbol(self):
        return self.loss1._symbol / self.loss2._symbol

    def _get_eval(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super()._get_eval(x_dict, **kwargs)
        return loss1 / loss2, x_dict


class LossSelfOperator(Loss):
    def __init__(self, loss1):
        _input_var = []

        if isinstance(loss1, type(None)):
            raise ValueError()

        if isinstance(loss1, Loss):
            _input_var = deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        else:
            raise ValueError()

        self._input_var = _input_var
        self.loss1 = loss1

    def train(self, x_dict={}, **kwargs):
        return self.loss1.train(x_dict, **kwargs)

    def test(self, x_dict={}, **kwargs):
        return self.loss1.test(x_dict, **kwargs)


class NegLoss(LossSelfOperator):
    """
    Apply the `neg` operation to the loss.

    Examples
    --------
    >>> loss_cls_1 = Parameter("x")
    >>> loss_cls = -loss_cls_1  # equals to NegLoss(loss_cls_1)
    >>> print(loss_cls)
    - x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    -4

    """
    @property
    def _symbol(self):
        return -self.loss1._symbol

    def _get_eval(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1._get_eval(x_dict, **kwargs)
        return -loss, x_dict


class AbsLoss(LossSelfOperator):
    """
    Apply the `abs` operation to two losses.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...            features_shape=[10])
    >>> loss_cls = LogProb(p).abs() # equals to AbsLoss(LogProb(p))
    >>> print(loss_cls)
    |\\log p(x)|
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([12.9894, 15.5280])

    """
    @property
    def _symbol(self):
        return sympy.Symbol("|{}|".format(self.loss1.loss_text))

    def _get_eval(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1._get_eval(x_dict, **kwargs)
        return loss.abs(), x_dict


class BatchMean(LossSelfOperator):
    r"""
    Average a loss class over given batch data.

    .. math::

        \mathbb{E}_{p_{data}(x)}[\mathcal{L}(x)] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...            features_shape=[10])
    >>> loss_cls = LogProb(p).mean() # equals to BatchMean(LogProb(p))
    >>> print(loss_cls)
    mean \left(\log p(x) \right)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(-14.5038)
    """

    @property
    def _symbol(self):
        return sympy.Symbol("mean \\left({} \\right)".format(self.loss1.loss_text))  # TODO: fix it

    def _get_eval(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1._get_eval(x_dict, **kwargs)
        return loss.mean(), x_dict


class BatchSum(LossSelfOperator):
    r"""
    Summation a loss class over given batch data.

    .. math::

        \sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...            features_shape=[10])
    >>> loss_cls = LogProb(p).sum() # equals to BatchSum(LogProb(p))
    >>> print(loss_cls)
    sum \left(\log p(x) \right)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(-31.9434)
    """

    @property
    def _symbol(self):
        return sympy.Symbol("sum \\left({} \\right)".format(self.loss1.loss_text))  # TODO: fix it

    def _get_eval(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1._get_eval(x_dict, **kwargs)
        return loss.sum(), x_dict


class SetLoss(Loss):
    def __init__(self, loss):
        self.loss = loss
        self._input_var = loss.input_var

    def __getattr__(self, name):
        getattr(self.loss, name)

    def _get_eval(self, x_dict, **kwargs):
        return self.loss._get_eval(x_dict, **kwargs)

    @property
    def _symbol(self):
        return self.loss._symbol


class Expectation(Loss):
    r"""
    Expectation of a given function (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^L f(x_l),

    where :math:`x_l \sim p(x)`.

    Note that :math:`f` doesn't need to be able to sample, which is known as the law of the unconscious statistician
    (LOTUS).

    Therefore, in this class, :math:`f` is assumed to :attr:`pixyz.Loss`.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"],
    ...            features_shape=[10]) # q(z|x)
    >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"],
    ...            features_shape=[10]) # p(x|z)
    >>> loss_cls = LogProb(p).expectation(q) # equals to Expectation(q, LogProb(p))
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) \right]
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([-12.8181, -12.6062])
    >>> loss_cls = LogProb(p).expectation(q, sample_shape=(5,)) # equals to Expectation(q, LogProb(p))
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP

    """

    def __init__(self, p, f, input_var=None, sample_shape=torch.Size([1])):

        if input_var is None:
            input_var = list(set(p.input_var) | set(f.input_var) - set(p.var))
        self._f = f
        self.sample_shape = torch.Size(sample_shape)

        super().__init__(p, input_var=input_var)

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol("\\mathbb{{E}}_{} \\left[{} \\right]".format(p_text, self._f.loss_text))

    def _get_eval(self, x_dict={}, **kwargs):
        samples_dicts = [self.p.sample(x_dict, reparam=True, return_all=True) for i in range(self.sample_shape.numel())]

        loss_and_dicts = [self._f.eval(samples_dict, return_dict=True, **kwargs) for
                          samples_dict in samples_dicts]  # TODO: eval or _get_eval
        losses = [loss for loss, loss_sample_dict in loss_and_dicts]
        # sum over sample_shape
        loss = torch.stack(losses).mean(dim=0)
        samples_dicts[0].update(loss_and_dicts[0][1])

        return loss, samples_dicts[0]
