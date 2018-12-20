from copy import deepcopy

from .losses import Loss
from ..utils import get_dict_values


class ARLoss(Loss):
    r"""
    Auto-regressive loss.

    This loss performs "scan-like" operation. You can implement any auto-regressive models
    by overriding this class.
    """

    def __init__(self, step_loss, last_loss=None,
                 step_fn=lambda x: x, max_iter=1, return_params=False,
                 initial_states={},
                 input_var=None):
        self.last_loss = last_loss
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.step_fn = step_fn
        self.initial_states = initial_states
        self.return_params = return_params

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            if self.last_loss is not None:
                _input_var += deepcopy(self.last_loss.input_var)
            if self.step_loss is not None:
                _input_var += deepcopy(self.step_loss.input_var)
            self._input_var = sorted(set(_input_var), key=_input_var.index)

    @property
    def loss_text(self):
        _loss_text = []
        if self.last_loss is not None:
            _loss_text.append(self.last_loss.loss_text)

        if self.step_loss is not None:
            _step_loss_text = "sum_(t=1)^(T={}) {}".format(str(self.max_iter),
                                                           self.step_loss.loss_text)
            _loss_text.append(_step_loss_text)

        return " + ".join(_loss_text)

    def estimate(self, x={}):
        x.update(self.initial_states)
        return x


class ARDRAWLoss(ARLoss):
    r"""
    Auto-regressive loss whose inputs are non-series data.

    .. math::

        \mathcal{L} = \mathcal{L}_{last}(x, h_T) + \sum_{t=1}^{T}\mathcal{L}_{step}(x, h_t),

    where :math:`h_t = f_{step}(h_{t-1}, x)`.
    """

    def __init__(self, step_loss, last_loss=None,
                 step_fn=lambda x: x, max_iter=1, return_params=False,
                 initial_states={},
                 input_var=None):

        super().__init__(step_loss, last_loss,
                         step_fn, max_iter, return_params,
                         initial_states, input_var)

    def estimate(self, x={}):
        x = super().estimate(x)

        step_loss_sum = 0
        for i in range(self.max_iter):
            step_loss_sum += self.step_loss.estimate(x)
            x = self.step_fn(i, x)
        loss = step_loss_sum + self.last_loss.estimate(x)

        if self.return_params:
            return loss, x

        return loss


class AutoRegressiveSeriesLoss(ARLoss):
    r"""
    Auto-regressive loss whose inputs are series data.

    .. math::

        \mathcal{L} = \mathcal{L}_{last}(x_1, h_T) + \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

    where :math:`h_t = f_{step}(h_{t-1}, x_{t-1})`.
    """

    def __init__(self, step_loss, last_loss=None,
                 step_fn=lambda x: x, max_iter=1, return_params=False,
                 initial_states={}, series_var=None,
                 input_var=None):

        super().__init__(step_loss, last_loss,
                         step_fn, max_iter, return_params,
                         initial_states, input_var)
        self.series_var = series_var
        self.non_series_var = list(set(self.input_var) - set(self.series_var))

    def select_step_inputs(self, i, x):
        x = get_dict_values(x, self.series_var, return_dict=True)
        return {k: v[i] for k, v in x.items()}

    def estimate(self, x={}):
        x = super().estimate(x)
        # TODO: finish to write this estimate method (unfinished for now)

        step_loss_sum = 0
        for i in range(self.max_iter):
            non_series_x = get_dict_values(step_x, self.non_series_var, return_dict=True)
            step_x = self.select_step_inputs(step_x)
            step_x.update(non_series_x)

            step_loss_sum += self.step_loss.estimate(x)
            x = self.step_fn(i, x)
        loss = step_loss_sum + self.last_loss.estimate(x)

        if self.return_params:
            return loss, x

        return loss


