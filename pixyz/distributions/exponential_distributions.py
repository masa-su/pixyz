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


class Normal(DistributionBase):
    """Normal distribution parameterized by :attr:`loc` and :attr:`scale`. """

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
        return NormalTorch

    @property
    def distribution_name(self):
        return "Normal"


class Bernoulli(DistributionBase):
    """Bernoulli distribution parameterized by :attr:`probs`."""

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
        return BernoulliTorch

    @property
    def distribution_name(self):
        return "Bernoulli"


class RelaxedBernoulli(Bernoulli):
    """Relaxed (re-parameterizable) Bernoulli distribution parameterized by :attr:`probs`."""

    def __init__(self, temperature=torch.tensor(0.1), cond_var=[], var=["x"], name="p", features_shape=torch.Size(),
                 **kwargs):
        self._temperature = temperature

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @property
    def distribution_torch_class(self):
        return BernoulliTorch

    @property
    def relaxed_distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedBernoulliTorch

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def set_dist(self, x_dict={}, sampling=True, batch_n=None, **kwargs):
        params = self.get_params(x_dict)
        if sampling is True:
            self._dist = self.relaxed_distribution_torch_class(temperature=self.temperature, **params)
        else:
            self._dist = self.distribution_torch_class(**params)

        # expand batch_n
        if batch_n:
            batch_shape = self._dist.batch_shape
            if batch_shape[0] == 1:
                self._dist = self._dist.expand(torch.Size([batch_n]) + batch_shape[1:])
            elif batch_shape[0] == batch_n:
                return
            else:
                raise ValueError()


class FactorizedBernoulli(Bernoulli):
    """
    Factorized Bernoulli distribution parameterized by :attr:`probs`.

    References
    ----------
    [Vedantam+ 2017] Generative Models of Visually Grounded Imagination

    """

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

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
        return CategoricalTorch

    @property
    def distribution_name(self):
        return "Categorical"


class RelaxedCategorical(Categorical):
    """Relaxed (re-parameterizable) categorical distribution parameterized by :attr:`probs`."""

    def __init__(self, temperature=torch.tensor(0.1), cond_var=[], var=["x"], name="p", features_shape=torch.Size(),
                 **kwargs):
        self._temperature = temperature

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @property
    def distribution_torch_class(self):
        return CategoricalTorch

    @property
    def relaxed_distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedOneHotCategoricalTorch

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def set_dist(self, x_dict={}, sampling=True, batch_n=None, **kwargs):
        params = self.get_params(x_dict)
        if sampling is True:
            self._dist = self.relaxed_distribution_torch_class(temperature=self.temperature, **params)
        else:
            self._dist = self.distribution_torch_class(**params)

        # expand batch_n
        if batch_n:
            batch_shape = self._dist.batch_shape
            if batch_shape[0] == 1:
                self._dist = self._dist.expand(torch.Size([batch_n]) + batch_shape[1:])
            elif batch_shape[0] == batch_n:
                return
            else:
                raise ValueError()

    def sample_mean(self, x_dict={}):
        self.set_dist(x_dict, sampling=False)
        return self.dist.mean

    def sample_variance(self, x_dict={}):
        self.set_dist(x_dict, sampling=False)
        return self.dist.variance


class Multinomial(DistributionBase):
    """Multinomial distribution parameterized by :attr:`total_count` and :attr:`probs`."""

    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), total_count=1, **kwargs):
        self._total_count = total_count

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, **kwargs)

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


class Dirichlet(DistributionBase):
    """Dirichlet distribution parameterized by :attr:`concentration`."""

    @property
    def params_keys(self):
        return ["concentration"]

    @property
    def distribution_torch_class(self):
        return DirichletTorch

    @property
    def distribution_name(self):
        return "Dirichlet"


class Beta(DistributionBase):
    """Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`."""

    @property
    def params_keys(self):
        return ["concentration1", "concentration0"]

    @property
    def distribution_torch_class(self):
        return BetaTorch

    @property
    def distribution_name(self):
        return "Beta"


class Laplace(DistributionBase):
    """
    Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
    """

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
        return LaplaceTorch

    @property
    def distribution_name(self):
        return "Laplace"


class Gamma(DistributionBase):
    """
    Gamma distribution parameterized by :attr:`concentration` and :attr:`rate`.
    """

    @property
    def params_keys(self):
        return ["concentration", "rate"]

    @property
    def distribution_torch_class(self):
        return GammaTorch

    @property
    def distribution_name(self):
        return "Gamma"
