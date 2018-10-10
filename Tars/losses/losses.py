class Loss(object):
    def __init__(self):
        pass

    def __add__(self, other):
        return AddLoss(self, other)

    def __sub__(self, other):
        return SubLoss(self, other)

    def __mul__(self, other):
        return MulLoss(self, other)

    def __truediv__(self, other):
        return DivLoss(self, other)

    def estimate(self):
        pass


class LossOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def estimate(self, x, **kwargs):
        if hasattr(self.a, "estimate"):
            a_estimated = self.a.estimate(x, **kwargs)
        else:
            a_estimated = self.a  # just a value

        if hasattr(self.b, "estimate"):
            b_estimated = self.b.estimate(x, **kwargs)
        else:
            b_estimated = self.b  # just a value

        return a_estimated, b_estimated


class AddLoss(LossOperator):
    def __init__(self, a, b):
        super(AddLoss, self).__init__(a, b)

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(AddLoss, self).estimate(x, **kwargs)

        return a_estimated + b_estimated


class SubLoss(LossOperator):
    def __init__(self, a, b):
        super(SubLoss, self).__init__(a, b)

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(SubLoss, self).estimate(x, **kwargs)

        return a_estimated - b_estimated


class MulLoss(LossOperator):
    def __init__(self, a, b):
        super(MulLoss, self).__init__(a, b)

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(MulLoss, self).estimate(x, **kwargs)

        return a_estimated * b_estimated


class DivLoss(LossOperator):
    def __init__(self, a, b):
        super(DivLoss, self).__init__(a, b)

    def estimate(self, x, **kwargs):
        a_estimated, b_estimated = \
            super(DivLoss, self).estimate(x, **kwargs)

        return a_estimated / b_estimated
