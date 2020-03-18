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

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["probs", "temperature"]

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedBernoulliTorch

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        raise NotImplementedError()

    def sample_mean(self, x_dict={}):
        raise NotImplementedError()

    def sample_variance(self, x_dict={}):
        raise NotImplementedError()

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

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ['probs', 'temperature']

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedOneHotCategoricalTorch

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        raise NotImplementedError()

    def sample_mean(self, x_dict={}):
        raise NotImplementedError()

    def sample_variance(self, x_dict={}):
        raise NotImplementedError()

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

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["concentration"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["concentration1", "concentration0"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
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

    @property
    def params_keys(self):
        return ["concentration", "rate"]

    @property
    def distribution_torch_class(self):
        return GammaTorch

    @property
    def distribution_name(self):
        return "Gamma"

    @property
    def has_reparam(self):
        return True
