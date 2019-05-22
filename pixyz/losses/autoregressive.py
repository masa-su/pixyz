from copy import deepcopy
import sympy

from .losses import Loss
from ..utils import get_dict_values
from sympy import Function

class IterativeLoss(Loss):
    r"""
    Iterative loss.

    This class allows implementing an arbitrary model which requires iteration (e.g., auto-regressive models).

    .. math::

        \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

    where :math:`x_t = f_{slice_step}(x, t)`.
    """

    def __init__(self, step_loss, max_iter=1,
                 input_var=None, series_var=None, update_value={}, slice_step=None, timestep_var=["t"]):
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.update_value = update_value
        self.timestep_var = timestep_var
        self.timpstep_symbol = sympy.Symbol(self.timestep_var[0])

        self.slice_step = slice_step
        if self.slice_step:
            self.step_loss = self.step_loss.expectation(self.slice_step)

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            _input_var += deepcopy(self.step_loss.input_var)

            self._input_var = sorted(set(_input_var), key=_input_var.index)

            if slice_step:
                self._input_var.remove(timestep_var[0])  # delete a time-step variable from input_var

        self.series_var = series_var

    @property
    def symbol(self):
        # TODO: naive implementation
        dummy_loss = sympy.Symbol("dummy_loss")
        _symbol = sympy.Sum(dummy_loss, (self.timpstep_symbol, 1, self.max_iter))
        _symbol = _symbol.subs({dummy_loss: self.step_loss.symbol})
        return _symbol

    def slice_step_fn(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def _get_eval(self, x, **kwargs):
        series_x = get_dict_values(x, self.series_var, return_dict=True)
        step_loss_sum = 0

        if "max_iter" in kwargs.keys():
            max_iter = kwargs["max_iter"]
        else:
            max_iter = self.max_iter

        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else:
            mask = None

        for t in range(max_iter):
            if self.slice_step:
                x.update({self.timestep_var[0]: t})
            else:
                # update series inputs & use slice_step_fn
                x.update(self.slice_step_fn(t, series_x))

            # evaluate
            step_loss, samples = self.step_loss.eval(x, return_dict=True)
            x.update(samples)
            if mask is not None:
                step_loss *= mask[t]
            step_loss_sum += step_loss

            # update
            for key, value in self.update_value.items():
                x.update({value: x[key]})

        loss = step_loss_sum

        x.update(series_x)
        return loss, x
