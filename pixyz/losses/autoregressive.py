from copy import deepcopy

from .losses import Loss
from ..utils import get_dict_values


class AutoRegressiveLoss(Loss):
    r"""
    Auto-regressive loss.

    This loss performs "scan-like" operation. You can implement any auto-regressive models
    by overriding this class.
    """

    def __init__(self, sequential_loss, static_loss=None,
                 step_func=lambda x: x, max_iter=1, return_params=False,
                 initial_states={},
                 input_var=None):
        self.static_loss = static_loss
        self.sequential_loss = sequential_loss
        self.max_iter = max_iter
        self.step_func = step_func
        self.initial_states = initial_states
        self.return_params = return_params

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            if self.static_loss is not None:
                _input_var += deepcopy(self.static_loss.input_var)
            if self.sequential_loss is not None:
                _input_var += deepcopy(self.sequential_loss.input_var)
            self._input_var = sorted(set(_input_var), key=_input_var.index)

    @property
    def loss_text(self):
        _loss_text = []
        if self.static_loss is not None:
            _loss_text.append(self.static_loss.loss_text)

        if self.sequential_loss is not None:
            _sequential_loss_text = "sum_(t=1)^(T={}) {}".format(str(self.max_iter),
                                                                 self.sequential_loss.loss_text)
            _loss_text.append(_sequential_loss_text)

        return " + ".join(_loss_text)

    def estimate(self, x={}):
        x.update(self.initial_states)
        return x


class AutoRegressiveDRAWLoss(AutoRegressiveLoss):
    r"""
    Auto-regressive loss whose inputs are non-series data.

    .. math::

        \mathcal{L} = \mathcal{L}_{static}(x, h_T) + \sum_{t=1}^{T}\mathcal{L}_{sequential}(x, h_t),

    where :math:`h_t = f_{step}(h_{t-1}, x)`.
    """

    def __init__(self, sequential_loss, static_loss=None,
                 step_func=lambda x: x, max_iter=1, return_params=False,
                 initial_states={},
                 input_var=None):

        super().__init__(sequential_loss, static_loss,
                         step_func, max_iter, return_params,
                         initial_states, input_var)

    def estimate(self, x={}):
        x = super().estimate(x)

        sequential_loss_sum = 0
        for i in range(self.max_iter):
            sequential_loss_sum += self.sequential_loss.estimate(x)
            x = self.step_func(i, x)
        loss = sequential_loss_sum + self.static_loss.estimate(x)

        if self.return_params:
            return loss, x

        return loss


class AutoRegressiveSeriesLoss(AutoRegressiveLoss):
    r"""
    Auto-regressive loss whose inputs are series data.

    .. math::

        \mathcal{L} = \mathcal{L}_{static}(x_1, h_T) + \sum_{t=1}^{T}\mathcal{L}_{sequential}(x_t, h_t),

    where :math:`h_t = f_{step}(h_{t-1}, x_{t-1})`.
    """

    def __init__(self, sequential_loss, static_loss=None,
                 step_func=lambda x: x, max_iter=1, return_params=False,
                 initial_states={}, series_var=None,
                 input_var=None):

        super().__init__(sequential_loss, static_loss,
                         step_func, max_iter, return_params,
                         initial_states, input_var)
        self.series_var = series_var
        self.static_var = list(set(self.input_var) - set(self.series_var))

    def select_step_inputs(self, i, x):
        x = get_dict_values(x, self.series_var, return_dict=True)
        return {k: v[i] for k, v in x.items()}

    def estimate(self, x={}):
        x = super().estimate(x)
        # TODO: finish to write estimate

        sequential_loss_sum = 0
        for i in range(self.max_iter):
            static_x = get_dict_values(step_x, self.static_var, return_dict=True)
            step_x = self.select_step_inputs(step_x)
            step_x.update(static_x)

            sequential_loss_sum += self.sequential_loss.estimate(x)
            x = self.step_func(i, x)
        loss = sequential_loss_sum + self.static_loss.estimate(x)

        if self.return_params:
            return loss, x

        return loss


