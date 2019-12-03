import torch
from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import RelaxedBernoulli as RelaxedBernoulliTorch
from torch.distributions import RelaxedOneHotCategorical as RelaxedOneHotCategoricalTorch
from torch.distributions.one_hot_categorical import OneHotCategorical as CategoricalTorch
from torch.distributions import Multinomial as MultinomialTorch
from torch.distributions import Dirichlet as DirichletTorch
from torch.distributions import Beta as BetaTorch
from torch.distributions import Laplace as LaplaceTorch
from torch.distributions import Gamma as GammaTorch

from ..utils import get_dict_values, sum_samples
from .distributions import DistributionBase


def _valid_param_dict(raw_dict):
    return {var_name: value for var_name, value in raw_dict.items() if value is not None}


class Normal(DistributionBase):
    """Normal distribution parameterized by :attr:`loc` and :attr:`scale`. """
    def __init__(self, cond_var=[], var=['x'], name='p', features_shape=torch.Size(), loc=None, scale=None):
        super().__init__(cond_var, var, name, features_shape, **_valid_param_dict({'loc': loc, 'scale': scale}))

    def get_params_keys(self, **kwargs):
        return ["loc", "scale"]

    def get_distribution_torch_class(self, **kwargs):
        return NormalTorch

    @property
    def distribution_name(self):
        return "Normal"

    @property
    def has_reparam(self):
        return True


class Bernoulli(DistributionBase):
    """Bernoulli distribution parameterized by :attr:`probs`."""
    def __init__(self, cond_var=[], var=['x'], name='p', features_shape=torch.Size(), probs=None):
        super().__init__(cond_var, var, name, features_shape, **_valid_param_dict({'probs': probs}))

    def get_params_keys(self, **kwargs):
        return ["probs"]

    def get_distribution_torch_class(self, **kwargs):
        return BernoulliTorch

    @property
    def distribution_name(self):
        return "Bernoulli"

    @property
    def has_reparam(self):
        return False


class RelaxedBernoulli(Bernoulli):
    """Relaxed (re-parameterizable) Bernoulli distribution parameterized by :attr:`probs` and :attr:`temperature`."""
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), temperature=torch.tensor(0.1),
                 probs=None):
        super(Bernoulli, self).__init__(cond_var, var, name, features_shape, **_valid_param_dict({
            'probs': probs, 'temperature': temperature}))

    def get_params_keys(self, relaxing=True, **kwargs):
        if relaxing:
            return ["probs", "temperature"]
        else:
            return ["probs"]

    def get_distribution_torch_class(self, relaxing=True, **kwargs):
        """Use relaxed version only when sampling"""
        if relaxing:
            return RelaxedBernoulliTorch
        else:
            return BernoulliTorch

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def set_dist(self, x_dict={}, relaxing=True, batch_n=None, **kwargs):
        super().set_dist(x_dict, relaxing, batch_n, **kwargs)

    def sample_mean(self, x_dict={}):
        self.set_dist(x_dict, relaxing=False)
        return self.dist.mean

    def sample_variance(self, x_dict={}):
        self.set_dist(x_dict, relaxing=False)
        return self.dist.variance

    @property
    def has_reparam(self):
        return True


class FactorizedBernoulli(Bernoulli):
    """
    Factorized Bernoulli distribution parameterized by :attr:`probs`.

    References
    ----------
    [Vedantam+ 2017] Generative Models of Visually Grounded Imagination

    """
    def __init__(self, cond_var=[], var=['x'], name='p', features_shape=torch.Size(), probs=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, probs=probs)

    @property
    def distribution_name(self):
        return "FactorizedBernoulli"

    def get_log_prob(self, x_dict):
        log_prob = super().get_log_prob(x_dict, sum_features=False)
        [_x] = get_dict_values(x_dict, self._var)
        log_prob[_x == 0] = 0
        log_prob = sum_samples(log_prob)
        return log_prob


class Categorical(DistributionBase):
    """Categorical distribution parameterized by :attr:`probs`."""
    def __init__(self, cond_var=[], var=['x'], name='p', features_shape=torch.Size(), probs=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'probs': probs}))

    def get_params_keys(self, **kwargs):
        return ["probs"]

    def get_distribution_torch_class(self, **kwargs):
        return CategoricalTorch

    @property
    def distribution_name(self):
        return "Categorical"

    @property
    def has_reparam(self):
        return False


class RelaxedCategorical(Categorical):
    """
    Relaxed (re-parameterizable) categorical distribution parameterized by :attr:`probs` and :attr:`temperature`.
    Notes: a shape of temperature should contain the event shape of this Categorical distribution.
    """
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), temperature=torch.tensor(0.1),
                 probs=None):
        super(Categorical, self).__init__(cond_var, var, name, features_shape,
                                          **_valid_param_dict({'probs': probs, 'temperature': temperature}))

    def get_params_keys(self, relaxing=True, **kwargs):
        if relaxing:
            return ['probs', 'temperature']
        else:
            return ['probs']

    def get_distribution_torch_class(self, relaxing=True, **kwargs):
        """Use relaxed version only when sampling"""
        if relaxing:
            return RelaxedOneHotCategoricalTorch
        else:
            return CategoricalTorch

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def set_dist(self, x_dict={}, relaxing=True, batch_n=None, **kwargs):
        super().set_dist(x_dict, relaxing, batch_n, **kwargs)

    def sample_mean(self, x_dict={}):
        self.set_dist(x_dict, relaxing=False)
        return self.dist.mean

    def sample_variance(self, x_dict={}):
        self.set_dist(x_dict, relaxing=False)
        return self.dist.variance

    @property
    def has_reparam(self):
        return True


class Multinomial(DistributionBase):
    """Multinomial distribution parameterized by :attr:`total_count` and :attr:`probs`."""

    def __init__(self, total_count=1, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), probs=None):
        self._total_count = total_count

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'probs': probs}))

    @property
    def total_count(self):
        return self._total_count

    def get_params_keys(self, **kwargs):
        return ["probs"]

    def get_distribution_torch_class(self, **kwargs):
        return MultinomialTorch

    @property
    def distribution_name(self):
        return "Multinomial"

    @property
    def has_reparam(self):
        return False


class Dirichlet(DistributionBase):
    """Dirichlet distribution parameterized by :attr:`concentration`."""
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), concentration=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration': concentration}))

    def get_params_keys(self, **kwargs):
        return ["concentration"]

    def get_distribution_torch_class(self, kwargs):
        return DirichletTorch

    @property
    def distribution_name(self):
        return "Dirichlet"

    @property
    def has_reparam(self):
        return True


class Beta(DistributionBase):
    """Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`."""
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(),
                 concentration1=None, concentration0=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration1': concentration1, 'concentration0': concentration0}))

    def get_params_keys(self, **kwargs):
        return ["concentration1", "concentration0"]

    def get_distribution_torch_class(self, **kwargs):
        return BetaTorch

    @property
    def distribution_name(self):
        return "Beta"

    @property
    def has_reparam(self):
        return True


class Laplace(DistributionBase):
    """
    Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
    """
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), loc=None, scale=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'loc': loc, 'scale': scale}))

    def get_params_keys(self, **kwargs):
        return ["loc", "scale"]

    def get_distribution_torch_class(self, **kwargs):
        return LaplaceTorch

    @property
    def distribution_name(self):
        return "Laplace"

    @property
    def has_reparam(self):
        return True


class Gamma(DistributionBase):
    """
    Gamma distribution parameterized by :attr:`concentration` and :attr:`rate`.
    """
    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), concentration=None, rate=None):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration': concentration, 'rate': rate}))

    def get_params_keys(self, **kwargs):
        return ["concentration", "rate"]

    def get_distribution_torch_class(self, **kwargs):
        return GammaTorch

    @property
    def distribution_name(self):
        return "Gamma"

    @property
    def has_reparam(self):
        return True
