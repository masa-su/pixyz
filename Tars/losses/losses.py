from ..utils import get_dict_values
import numbers


class Loss(object):
    def __init__(self, p, q=None, input_var=[]):
        if len(input_var) > 0:
            self.input_var = input_var
        else:
            _input_var = p.cond_var
            if q is not None:
                _input_var += q.cond_var
                _input_var = sorted(set(_input_var), key=_input_var.index)
            self.input_var = _input_var

        self.loss_text = "loss({},{})".format(p, q)

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
        return MulLoss(self, -1)

    def mean(self):
        return BatchMean(self)

    def sum(self):
        return BatchSum(self)

    def estimate(self, x):
        return get_dict_values(x, self.input_var, True)


class LossOperator(Loss):
    def __init__(self, a, b):
        _input_var = []
        if hasattr(a, "input_var"):
            _input_var += a.input_var

        if hasattr(b, "input_var"):
            _input_var += b.input_var

        _input_var = sorted(set(_input_var), key=_input_var.index)
        self.input_var = _input_var

        self.a = a
        self.b = b

    def estimate(self, x, **kwargs):
        if hasattr(self.a, "estimate"):
            a_estimated = self.a.estimate(x, **kwargs)
        else:
            if isinstance(self.a, numbers.Number):
                a_estimated = self.a  # just a value
            elif isinstance(self.a, type(None)):
                a_estimated = 0
            else:
                raise ValueError

        if hasattr(self.b, "estimate"):
            b_estimated = self.b.estimate(x, **kwargs)
        else:
            if isinstance(self.b, numbers.Number):
                b_estimated = self.b  # just a value
            elif isinstance(self.b, type(None)):
                b_estimated = 0
            else:
                raise ValueError

        return a_estimated, b_estimated


class AddLoss(LossOperator):
    def __init__(self, a, b):
        super(AddLoss, self).__init__(a, b)
        self.loss_text = "{} + {}".format(str(a), str(b))

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(AddLoss, self).estimate(x, **kwargs)

        return a_estimated + b_estimated


class SubLoss(LossOperator):
    def __init__(self, a, b):
        super(SubLoss, self).__init__(a, b)
        self.loss_text = "{} - {}".format(str(a), str(b))

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(SubLoss, self).estimate(x, **kwargs)

        return a_estimated - b_estimated


class MulLoss(LossOperator):
    def __init__(self, a, b):
        super(MulLoss, self).__init__(a, b)
        self.loss_text = "{} * {}".format(str(a), str(b))

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(MulLoss, self).estimate(x, **kwargs)

        return a_estimated * b_estimated


class DivLoss(LossOperator):
    def __init__(self, a, b):
        super(DivLoss, self).__init__(a, b)
        self.loss_text = "{} / {}".format(str(a), str(b))

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(DivLoss, self).estimate(x, **kwargs)

        return a_estimated / b_estimated


class LossSelfOperator(Loss):
    def __init__(self, a):
        self.input_var = a
        self.a = a


class BatchMean(LossSelfOperator):
    def __init__(self, a):
        super(BatchMean, self).__init__(a)
        self.loss_text = str(a)  # TODO: fix it

    def estimate(self, x, **kwargs):
        loss = self.a.estimate(x, **kwargs)
        return loss.mean()


class BatchSum(LossSelfOperator):
    def __init__(self, a):
        super(BatchSum, self).__init__(a)
        self.loss_text = str(a)  # TODO: fix it

    def estimate(self, x, **kwargs):
        loss = self.a.estimate(x, **kwargs)
        return loss.sum()
