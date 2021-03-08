from copy import deepcopy
import sympy

from .losses import Loss
from ..utils import get_dict_values, replace_dict_keys


class IterativeLoss(Loss):
    r"""
    Iterative loss.

    This class allows implementing an arbitrary model which requires iteration.

    .. math::

        \mathcal{L} = \sum_{t=0}^{T-1}\mathcal{L}_{step}(x_t, h_t),

    where :math:`x_t = f_{slice\_step}(x, t)`.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal, Bernoulli, Deterministic
    >>>
    >>> # Set distributions
    >>> x_dim = 128
    >>> z_dim = 64
    >>> h_dim = 32
    >>>
    >>> # p(x|z,h_{prev})
    >>> class Decoder(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(var=["x"],cond_var=["z", "h_prev"],name="p")
    ...         self.fc = torch.nn.Linear(z_dim + h_dim, x_dim)
    ...     def forward(self, z, h_prev):
    ...         return {"probs": torch.sigmoid(self.fc(torch.cat((z, h_prev), dim=-1)))}
    ...
    >>> # q(z|x,h_{prev})
    >>> class Encoder(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x", "h_prev"],name="q")
    ...         self.fc_loc = torch.nn.Linear(x_dim + h_dim, z_dim)
    ...         self.fc_scale = torch.nn.Linear(x_dim + h_dim, z_dim)
    ...     def forward(self, x, h_prev):
    ...         xh = torch.cat((x, h_prev), dim=-1)
    ...         return {"loc": self.fc_loc(xh), "scale": F.softplus(self.fc_scale(xh))}
    ...
    >>> # f(h|x,z,h_{prev}) (update h)
    >>> class Recurrence(Deterministic):
    ...     def __init__(self):
    ...         super().__init__(var=["h"], cond_var=["x", "z", "h_prev"], name="f")
    ...         self.rnncell = torch.nn.GRUCell(x_dim + z_dim, h_dim)
    ...     def forward(self, x, z, h_prev):
    ...         return {"h": self.rnncell(torch.cat((z, x), dim=-1), h_prev)}
    >>>
    >>> p = Decoder()
    >>> q = Encoder()
    >>> f = Recurrence()
    >>>
    >>> # Set the loss class
    >>> step_loss_cls = p.log_prob().expectation(q * f).mean()
    >>> print(step_loss_cls)
    mean \left(\mathbb{E}_{q(z,h|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
    >>> loss_cls = IterativeLoss(step_loss=step_loss_cls,
    ...                          series_var=["x"], update_value={"h": "h_prev"})
    >>> print(loss_cls)
    \sum_{t=0}^{-1 + t_{max}} mean \left(\mathbb{E}_{q(z,h|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
    >>>
    >>> # Evaluate
    >>> x_sample = torch.randn(30, 2, 128) # (timestep_size, batch_size, feature_size)
    >>> h_init = torch.zeros(2, 32) # (batch_size, h_dim)
    >>> loss = loss_cls.eval({"x": x_sample, "h_prev": h_init})
    >>> print(loss) # doctest: +SKIP
    tensor(-2826.0906, grad_fn=<AddBackward0>
    """

    def __init__(self, step_loss, max_iter=None,
                 series_var=(), update_value={}, slice_step=None, timestep_var=()):
        super().__init__()
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.update_value = update_value
        self.timestep_var = timestep_var
        if timestep_var:
            self.timpstep_symbol = sympy.Symbol(self.timestep_var[0])
        else:
            self.timpstep_symbol = sympy.Symbol("t")

        if not series_var and (max_iter is None):
            raise ValueError()

        self.slice_step = slice_step
        if self.slice_step:
            self.step_loss = self.step_loss.expectation(self.slice_step)

        _input_var = []
        _input_var += deepcopy(self.step_loss.input_var)
        _input_var += series_var
        _input_var += update_value.values()

        self._input_var = sorted(set(_input_var), key=_input_var.index)

        if timestep_var:
            self._input_var.remove(timestep_var[0])  # delete a time-step variable from input_var

        self.series_var = series_var

    @property
    def _symbol(self):
        # TODO: naive implementation
        dummy_loss = sympy.Symbol("dummy_loss")
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = sympy.Symbol(sympy.latex(self.timpstep_symbol, order="old", mode="plain") + "_{max}")

        _symbol = sympy.Sum(dummy_loss, (self.timpstep_symbol, 0, max_iter - 1))
        _symbol = _symbol.subs({dummy_loss: self.step_loss._symbol})
        return _symbol

    def slice_step_fn(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def forward(self, x_dict, **kwargs):
        series_x_dict = get_dict_values(x_dict, self.series_var, return_dict=True)
        updated_x_dict = get_dict_values(x_dict, list(self.update_value.values()), return_dict=True)

        step_loss_sum = 0

        # set max_iter
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = len(series_x_dict[self.series_var[0]])

        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else:
            mask = None

        for t in range(max_iter):
            if self.timestep_var:
                x_dict.update({self.timestep_var[0]: t})
            if not self.slice_step:
                # update series inputs & use slice_step_fn
                x_dict.update(self.slice_step_fn(t, series_x_dict))

            # evaluate
            step_loss, samples = self.step_loss.eval(x_dict, return_dict=True, return_all=False)
            x_dict.update(samples)
            if mask is not None:
                step_loss *= mask[t]
            step_loss_sum += step_loss

            # update
            x_dict = replace_dict_keys(x_dict, self.update_value)

        loss = step_loss_sum

        # Restore original values
        x_dict.update(series_x_dict)
        x_dict.update(updated_x_dict)
        # TODO: x_dict contains no-updated variables.
        return loss, x_dict
