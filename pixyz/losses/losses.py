import abc
import sympy

import numbers
from copy import deepcopy

from ..utils import tolist


class Loss(object, metaclass=abc.ABCMeta):
    def __init__(self, p, q=None, input_var=None):
        self._p = p
        self._q = q

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
        return self._input_var

    @property
    @abc.abstractmethod
    def _symbol(self):
        raise NotImplementedError

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
        return AddLoss(self, other)

    def __sub__(self, other):
        return SubLoss(self, other)

    def __rsub__(self, other):
        return SubLoss(self, other)

    def __mul__(self, other):
        return MulLoss(self, other)

    def __rmul__(self, other):
        return MulLoss(self, other)

    def __truediv__(self, other):
        return DivLoss(self, other)

    def __rtruediv__(self, other):
        return DivLoss(self, other)

    def __neg__(self):
        return NegLoss(self)

    def abs(self):
        return AbsLoss(self)

    def mean(self):
        return BatchMean(self)

    def sum(self):
        return BatchSum(self)

    def eval(self, x={}, return_dict=False, **kwargs):
        if not(set(list(x.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, got {}.".format(list(x.keys())))

        loss, x = self._get_eval(x, **kwargs)

        if return_dict:
            return loss, x

        return loss

    def expectation(self, p, input_var=None):
        return Expectation(p, self, input_var=input_var)

    @abc.abstractmethod
    def _get_eval(self, x, **kwargs):
        raise NotImplementedError


class ValueLoss(Loss):
    def __init__(self, loss1):
        self._loss1 = loss1
        self._input_var = []

    def _get_eval(self, x={}, **kwargs):
        return self._loss1, x

    @property
    def _symbol(self):
        return self._loss1


class Parameter(Loss):
    def __init__(self, input_var):
        if not isinstance(input_var, str):
            raise ValueError
        self._input_var = tolist(input_var)

    def _get_eval(self, x={}, **kwargs):
        return x[self._input_var[0]], x

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
        self._loss1 = loss1
        self._loss2 = loss2

    def _get_eval(self, x={}, **kwargs):
        if not isinstance(self._loss1, type(None)):
            loss1, x1 = self._loss1._get_eval(x, **kwargs)
        else:
            loss1 = 0
            x1 = {}

        if not isinstance(self._loss2, type(None)):
            loss2, x2 = self._loss2._get_eval(x, **kwargs)
        else:
            loss2 = 0
            x2 = {}

        x1.update(x2)

        return loss1, loss2, x1


class AddLoss(LossOperator):
    @property
    def _symbol(self):
        return self._loss1._symbol + self._loss2._symbol

    def _get_eval(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_eval(x, **kwargs)
        return loss1 + loss2, x


class SubLoss(LossOperator):
    @property
    def _symbol(self):
        return self._loss1._symbol - self._loss2._symbol

    def _get_eval(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_eval(x, **kwargs)
        return loss1 - loss2, x


class MulLoss(LossOperator):
    @property
    def _symbol(self):
        return self._loss1._symbol * self._loss2._symbol

    def _get_eval(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_eval(x, **kwargs)
        return loss1 * loss2, x


class DivLoss(LossOperator):
    @property
    def _symbol(self):
        return self._loss1._symbol / self._loss2._symbol

    def _get_eval(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_eval(x, **kwargs)
        return loss1 / loss2, x


class LossSelfOperator(Loss):
    def __init__(self, loss1):
        _input_var = []

        if isinstance(loss1, type(None)):
            raise ValueError

        if isinstance(loss1, Loss):
            _input_var = deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        else:
            raise ValueError

        self._input_var = _input_var
        self._loss1 = loss1

    def train(self, x={}, **kwargs):
        return self._loss1.train(x, **kwargs)

    def test(self, x={}, **kwargs):
        return self._loss1.test(x, **kwargs)


class NegLoss(LossSelfOperator):
    @property
    def _symbol(self):
        return -self._loss1._symbol

    def _get_eval(self, x={}, **kwargs):
        loss, x = self._loss1._get_eval(x, **kwargs)
        return -loss, x


class AbsLoss(LossSelfOperator):
    @property
    def _symbol(self):
        return sympy.Symbol("|{}|".format(self._loss1.loss_text))

    def _get_eval(self, x={}, **kwargs):
        loss, x = self._loss1._get_eval(x, **kwargs)
        return loss.abs(), x


class BatchMean(LossSelfOperator):
    r"""
    Loss averaged over batch data.

    .. math::

        \mathbb{E}_{p_{data}(x)}[\mathcal{L}(x)] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.
    """

    @property
    def _symbol(self):
        return sympy.Symbol("mean \\left({} \\right)".format(self._loss1.loss_text))  # TODO: fix it

    def _get_eval(self, x={}, **kwargs):
        loss, x = self._loss1._get_eval(x, **kwargs)
        return loss.mean(), x


class BatchSum(LossSelfOperator):
    r"""
    Loss summed over batch data.

    .. math::

        \sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.
    """

    @property
    def _symbol(self):
        return sympy.Symbol("sum \\left({} \\right)".format(self._loss1.loss_text))  # TODO: fix it

    def _get_eval(self, x={}, **kwargs):
        loss, x = self._loss1._get_eval(x, **kwargs)
        return loss.sum(), x


class SetLoss(Loss):
    def __init__(self, loss):
        self._loss = loss
        self._input_var = loss._input_var

    def __getattr__(self, name):
        getattr(self._loss, name)

    def _get_eval(self, x, **kwargs):
        return self._loss._get_eval(x, **kwargs)

    @property
    def _symbol(self):
        return self._loss._symbol


class Expectation(Loss):
    r"""
    Expectation of a given function (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^L f(x_l),

    where :math:`x_l \sim p(x)`.

    Note that :math:`f` doesn't need to be able to sample, which is known as the law of the unconscious statistician
    (LOTUS).

    Therefore, in this class, :math:`f` is assumed to :attr:`pixyz.Loss`.
    """

    def __init__(self, p, f, input_var=None):

        if input_var is None:
            input_var = list(set(p.input_var) | set(f.input_var) - set(p.var))
        self._f = f

        super().__init__(p, input_var=input_var)

    @property
    def _symbol(self):
        p_text = "{" + self._p.prob_text + "}"
        return sympy.Symbol("\\mathbb{{E}}_{} \\left[{} \\right]".format(p_text, self._f.loss_text))

    def _get_eval(self, x={}, **kwargs):
        samples_dict = self._p.sample(x, reparam=True, return_all=True)

        # TODO: whether eval or _get_eval
        loss, loss_sample_dict = self._f.eval(samples_dict, return_dict=True, **kwargs)
        samples_dict.update(loss_sample_dict)

        return loss, samples_dict
