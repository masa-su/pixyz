from __future__ import print_function

from .distributions import Distribution


class Deterministic(Distribution):
    """
    Deterministic distribution (or degeneration distribution)

    Examples
    --------
    >>> import torch
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super().__init__(var=["x"], cond_var=["z"])
    ...         self.model = torch.nn.Linear(64, 512)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p = Generator()
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=512, bias=True)
      )
    >>> sample = p.sample({"z": torch.randn(1, 64)})
    >>> p.log_prob().eval(sample) # log_prob is not defined.
    Traceback (most recent call last):
     ...
    NotImplementedError: Log probability of deterministic distribution is not defined.
    """

    def __init__(self, var, cond_var=[], name='p', **kwargs):
        super().__init__(var=var, cond_var=cond_var, name=name, **kwargs)

    @property
    def distribution_name(self):
        return "Deterministic"

    def sample(self, x_dict={}, return_all=True, **kwargs):
        input_dict = self._get_input_dict(x_dict)
        output_dict = self.forward(**input_dict)

        if set(output_dict.keys()) != set(self._var):
            raise ValueError("Output variables are not the same as `var`.")

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        raise NotImplementedError("Log probability of deterministic distribution is not defined.")

    @property
    def has_reparam(self):
        return True


class EmpiricalDistribution(Distribution):
    """
    Data distribution.

    Samples from this distribution equal given inputs.

    Examples
    --------
    >>> import torch
    >>> p = EmpiricalDistribution(var=["x"])
    >>> print(p)
    Distribution:
      p_{data}(x)
    Network architecture:
      EmpiricalDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> sample = p.sample({"x": torch.randn(1, 64)})
    """

    def __init__(self, var, name="p_{data}"):
        super().__init__(var=var, cond_var=[], name=name)

    @property
    def distribution_name(self):
        return "Data distribution"

    def sample(self, x_dict={}, return_all=True, **kwargs):
        output_dict = self._get_input_dict(x_dict)

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict
        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        raise NotImplementedError()

    @property
    def input_var(self):
        """
        In EmpiricalDistribution, `input_var` is same as `var`.
        """

        return self.var

    @property
    def has_reparam(self):
        return True
