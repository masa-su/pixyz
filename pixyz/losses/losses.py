import abc

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
    def loss_text(self):
        raise NotImplementedError

    def __str__(self):
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

    def estimate(self, x={}, return_dict=False, **kwargs):
        if not(set(list(x.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, got {}.".format(list(x.keys())))

        loss, x = self._get_estimated_value(x, **kwargs)

        if return_dict:
            return loss, x

        return loss

    @abc.abstractmethod
    def _get_estimated_value(self, x, **kwargs):
        raise NotImplementedError


class ValueLoss(Loss):
    def __init__(self, loss1):
        self._loss1 = loss1
        self._input_var = []

    def _get_estimated_value(self, x={}, **kwargs):
        return self._loss1, x

    @property
    def loss_text(self):
        return str(self._loss1)


class Parameter(Loss):
    def __init__(self, input_var):
        if not isinstance(input_var, str):
            raise ValueError
        self._input_var = tolist(input_var)

    def _get_estimated_value(self, x={}, **kwargs):
        return x[self._input_var[0]], x

    @property
    def loss_text(self):
        return str(self._input_var[0])


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

    @property
    def _loss_text_list(self):
        loss_text_list = []
        if not isinstance(self._loss1, type(None)):
            loss_text_list.append(self._loss1.loss_text)

        if not isinstance(self._loss2, type(None)):
            loss_text_list.append(self._loss2.loss_text)

        return loss_text_list

    @property
    def loss_text(self):
        raise NotImplementedError

    def _get_estimated_value(self, x={}, **kwargs):
        if not isinstance(self._loss1, type(None)):
            loss1, x1 = self._loss1._get_estimated_value(x, **kwargs)
        else:
            loss1 = 0
            x1 = {}

        if not isinstance(self._loss2, type(None)):
            loss2, x2 = self._loss2._get_estimated_value(x, **kwargs)
        else:
            loss2 = 0
            x2 = {}

        x1.update(x2)

        return loss1, loss2, x1

    def train(self, x, **kwargs):
        """
        TODO: Fix
        """
        loss1 = self._loss1.train(x, **kwargs)
        loss2 = self._loss2.train(x, **kwargs)

        return loss1 + loss2

    def test(self, x, **kwargs):
        """
        TODO: Fix
        """
        loss1 = self._loss1.test(x, **kwargs)
        loss2 = self._loss2.test(x, **kwargs)

        return loss1 + loss2


class AddLoss(LossOperator):
    @property
    def loss_text(self):
        return " + ".join(self._loss_text_list)

    def _get_estimated_value(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_estimated_value(x, **kwargs)
        return loss1 + loss2, x


class SubLoss(LossOperator):
    @property
    def loss_text(self):
        return " - ".join(self._loss_text_list)

    def _get_estimated_value(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_estimated_value(x, **kwargs)
        return loss1 - loss2, x


class MulLoss(LossOperator):
    @property
    def loss_text(self):
        return " * ".join(self._loss_text_list)

    def _get_estimated_value(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_estimated_value(x, **kwargs)
        return loss1 * loss2, x


class DivLoss(LossOperator):
    @property
    def loss_text(self):
        return " / ".join(self._loss_text_list)

    def _get_estimated_value(self, x={}, **kwargs):
        loss1, loss2, x = super()._get_estimated_value(x, **kwargs)
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
    def loss_text(self):
        return "-({})".format(self._loss1.loss_text)

    def _get_estimated_value(self, x={}, **kwargs):
        loss, x = self._loss1._get_estimated_value(x, **kwargs)
        return -loss, x


class AbsLoss(LossSelfOperator):
    @property
    def loss_text(self):
        return "|{}|".format(self._loss1.loss_text)

    def _get_estimated_value(self, x={}, **kwargs):
        loss, x = self._loss1._get_estimated_value(x, **kwargs)
        return loss.abs(), x


class BatchMean(LossSelfOperator):
    r"""
    Loss averaged over batch data.

    .. math::

        \mathbb{E}_{p_{data}(x)}[\mathcal{L}(x)] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.
    """

    @property
    def loss_text(self):
        return "mean({})".format(self._loss1.loss_text)  # TODO: fix it

    def _get_estimated_value(self, x={}, **kwargs):
        loss, x = self._loss1._get_estimated_value(x, **kwargs)
        return loss.mean(), x


class BatchSum(LossSelfOperator):
    r"""
    Loss summed over batch data.

    .. math::

        \sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.
    """

    @property
    def loss_text(self):
        return "sum({})".format(self._loss1.loss_text)  # TODO: fix it

    def _get_estimated_value(self, x={}, **kwargs):
        loss, x = self._loss1._get_estimated_value(x, **kwargs)
        return loss.sum(), x


class SetLoss(Loss):
    def __init__(self, loss):
        self._loss = loss
        self._input_var = loss._input_var

    def __getattr__(self, name):
        getattr(self._loss, name)

    def _get_estimated_value(self, x, **kwargs):
        return self._loss._get_estimated_value(x, **kwargs)

    @property
    def loss_text(self):
        return self._loss.loss_text
