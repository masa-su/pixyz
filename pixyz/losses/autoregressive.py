from copy import deepcopy

from .losses import Loss
from .expectations import LossExpectation
from ..utils import get_dict_values


class IterativeLoss(Loss):
    r"""
    Iterative loss.

    You can implement arbitrary model which needs to iteration (e.g., auto-regressive models) with this class.

    .. math::

        \mathcal{L} = \mathcal{L}_{last}(x_1, h_T) + \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),
    """

    def __init__(self, step_loss, last_loss=None, max_iter=1,
                 input_var=None, series_var=None, update_value={}, slice_step=None, timestep_var="t"):
        self.last_loss = last_loss
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.update_value = update_value
        self.timestep_var = timestep_var

        self.slice_step = slice_step
        if self.slice_step:
            self.step_loss = LossExpectation(self.slice_step, self.step_loss)

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            if self.last_loss is not None:
                _input_var += deepcopy(self.last_loss.input_var)
            if self.step_loss is not None:
                _input_var += deepcopy(self.step_loss.input_var)
            self._input_var = sorted(set(_input_var), key=_input_var.index)

            if self.step_loss:
                self._input_var.remove(timestep_var)  # delete the variable of time step

        self.series_var = series_var

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

    def slice_step_fn(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def _get_estimated_value(self, x, **kwargs):
        series_x = get_dict_values(x, self.series_var, return_dict=True)
        step_loss_sum = 0

        if "max_iter" in kwargs.keys():
            max_iter = kwargs["max_iter"]
        else: max_iter = self.max_iter

        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else: mask = None
        
        for t in range(max_iter):
            if self.slice_step:
                x.update({self.timestep_var: t})
            else:
                # update series inputs & use slice_step_fn
                x.update(self.slice_step_fn(t, series_x))

            # estimate
            step_loss, samples = self.step_loss.estimate(x, return_dict=True)
            x.update(samples)
            if mask is not None: step_loss *= mask[t]
            step_loss_sum += step_loss

            # update
            for key, value in self.update_value.items():
                x.update({value: x[key]})

        loss = step_loss_sum

        if self.last_loss is not None:
            x.update(self.slice_step_from_inputs(0, series_x))
            loss += self.last_loss.estimate(x)

        x.update(series_x)
        return loss, x
