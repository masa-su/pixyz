from copy import deepcopy

from .losses import Loss


class AutoRegressive(Loss):
    r"""
    Auto-regressive loss.

    .. math::

        \mathcal{L} = \mathcal{L}_{static}(x, h_T) + \sum_{t=1}^{T}\mathcal{L}_{sequential}(x, h_t),

    where :math:`h_t = f_{step}(h_{t-1}, x)`.
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

        sequential_loss_sum = 0
        for i in range(self.max_iter):

            sequential_loss_sum += self.sequential_loss.estimate(x)
            x = self.step_func(i, x)
        loss = sequential_loss_sum + self.static_loss.estimate(x)

        if self.return_params:
            return loss, x

        return loss

