from copy import deepcopy

from .losses import Loss
from ..utils import get_dict_values


class ARLoss(Loss):
    r"""
    Auto-regressive loss.

    This loss performs "scan-like" operation. You can implement any auto-regressive models
    by overriding this class.

    .. math::

        \mathcal{L} = \mathcal{L}_{last}(x_1, h_T) + \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

    where :math:`h_t = f_{step}(x_{t-1}, h_{t-1})`.
    """

    def __init__(self, step_loss, last_loss=None,
                 fn=lambda x: x, max_iter=1, return_params=False,
                 input_var=None, series_var=None, update_value=None):
        self.last_loss = last_loss
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.fn = fn
        self.return_params = return_params
        self.update_value = update_value

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            if self.last_loss is not None:
                _input_var += deepcopy(self.last_loss.input_var)
            if self.step_loss is not None:
                _input_var += deepcopy(self.step_loss.input_var)
            self._input_var = sorted(set(_input_var), key=_input_var.index)

        self.series_var = series_var
        self.non_series_var = list(set(self.input_var) - set(self.series_var))

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

    def slice_step_from_inputs(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def estimate(self, x={}):
        x = super().estimate(x)
        series_x = get_dict_values(x, self.series_var, return_dict=True)
        step_loss_sum = 0

        for t in range(self.max_iter):
            # update series inputs
            x.update(self.slice_step_from_inputs(t, series_x))

            # sample and step
            x = self.fn(t, **x)

            # estimate
            step_loss_sum += self.step_loss.estimate(x)

            # update
            for key, value in self.update_value.items():
                x.update({value: x[key]})

        loss = step_loss_sum

        if self.last_loss is not None:
            x.update(self.slice_step_from_inputs(0, series_x))
            loss += self.last_loss.estimate(x)

        if self.return_params:
            x.update(series_x)
            return loss, x

        return loss

