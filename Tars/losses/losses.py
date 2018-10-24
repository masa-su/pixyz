import numbers
from copy import deepcopy

from ..utils import get_dict_values


class Loss(object):
    def __init__(self, p, q=None, input_var=[]):
        if len(input_var) > 0:
            self.input_var = input_var
        else:
            _input_var = deepcopy(p.cond_var)
            if q is not None:
                _input_var += deepcopy(q.cond_var)
                _input_var = sorted(set(_input_var), key=_input_var.index)
                self.loss_text = "loss({},{})".format(p.prob_text, q.prob_text)
            self.input_var = _input_var

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

    def mean(self):
        return BatchMean(self)

    def sum(self):
        return BatchSum(self)

    def estimate(self, x):
        # if not set(list(x.keys())) == set(self.input_var):
        #     raise ValueError("Input's keys are not valid.")
        return get_dict_values(x, self.input_var, True)


class ValueLoss(Loss):
    def __init__(self, a):
        self.a = a
        self.input_var = []
        self.loss_text = str(a)

    def estimate(self, x, **kwargs):
        return self.a


class LossOperator(Loss):
    def __init__(self, a, b):
        _input_var = []
        _loss_text = []

        if not isinstance(a, type(None)):
            if isinstance(a, Loss):
                _input_var += deepcopy(a.input_var)
            elif isinstance(a, numbers.Number):
                a = ValueLoss(a)
            else:
                raise ValueError
            _loss_text.append(a.loss_text)

        if not isinstance(b, type(None)):
            if isinstance(b, Loss):
                _input_var += deepcopy(b.input_var)
            elif isinstance(b, numbers.Number):
                b = ValueLoss(b)
            else:
                raise ValueError
            _loss_text.append(b.loss_text)

        _input_var = sorted(set(_input_var), key=_input_var.index)

        self.input_var = _input_var
        self.a = a
        self.b = b

        if len(_loss_text) != 0:
            self.loss_text = ' {} '.join(_loss_text)
        else:
            raise ValueError

    def estimate(self, x, **kwargs):
        if not isinstance(self.a, type(None)):
            a_loss = self.a.estimate(x, **kwargs)
        else:
            a_loss = 0

        if not isinstance(self.b, type(None)):
            b_loss = self.b.estimate(x, **kwargs)
        else:
            b_loss = 0

        return a_loss, b_loss


class AddLoss(LossOperator):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.loss_text = self.loss_text.format("+")

    def estimate(self, x, **kwargs):
        a_loss, b_loss = \
            super().estimate(x, **kwargs)

        return a_loss + b_loss


class SubLoss(LossOperator):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.loss_text = self.loss_text.format("-")

    def estimate(self, x, **kwargs):
        a_loss, b_loss = \
            super().estimate(x, **kwargs)

        return a_loss - b_loss


class MulLoss(LossOperator):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.loss_text = self.loss_text.format("*")

    def estimate(self, x, **kwargs):
        a_loss, b_loss = \
            super().estimate(x, **kwargs)

        return a_loss * b_loss


class DivLoss(LossOperator):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.loss_text = self.loss_text.format("/")

    def estimate(self, x, **kwargs):
        a_loss, b_loss = \
            super().estimate(x, **kwargs)

        return a_loss / b_loss


class LossSelfOperator(Loss):
    def __init__(self, a):
        _loss_text = ""

        if not isinstance(a, type(None)):
            if isinstance(a, Loss):
                _input_var = deepcopy(a.input_var)
            elif isinstance(a, numbers.Number):
                a = ValueLoss(a)
            else:
                raise ValueError
            _loss_text += a.loss_text

        self.input_var = _input_var
        self.a = a
        self.loss_text = _loss_text


class NegLoss(LossSelfOperator):
    def __init__(self, a):
        super().__init__(a)
        self.loss_text = "- " + self.loss_text

    def estimate(self, x, **kwargs):
        loss = self.a.estimate(x, **kwargs)

        return -loss


class BatchMean(LossSelfOperator):
    def __init__(self, a):
        super().__init__(a)
        self.loss_text = a.loss_text  # TODO: fix it

    def estimate(self, x, **kwargs):
        loss = self.a.estimate(x, **kwargs)
        return loss.mean()


class BatchSum(LossSelfOperator):
    def __init__(self, a):
        super().__init__(a)
        self.loss_text = a.loss_text  # TODO: fix it

    def estimate(self, x, **kwargs):
        loss = self.a.estimate(x, **kwargs)
        return loss.sum()
