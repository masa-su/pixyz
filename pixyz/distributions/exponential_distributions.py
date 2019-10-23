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

from .sample_dict import SampleDict
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
        return ["probs", "logits"]

    @property
    def distribution_torch_class(self):
        return BernoulliTorch

    @property
    def distribution_name(self):
        return "Bernoulli"


class RelaxedBernoulli(Bernoulli):
    """Relaxed (re-parameterizable) Bernoulli distribution parameterized by :attr:`probs`."""

    def __init__(self, temperature=torch.tensor(0.1), cond_var=(), var=("x",), name="p", features_shape=torch.Size(),
                 **kwargs):
        self._temperature = temperature
        self.relaxing = False

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return BernoulliTorch if not self.relaxing else RelaxedBernoulliTorch

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def set_dist(self, x_dict=None, **dist_options):
        if self.relaxing:
            super().set_dist(x_dict, temperature=self.temperature, **dist_options)
        else:
            super().set_dist(x_dict, **dist_options)

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False, **kwargs):
        tmp = self.relaxing
        self.relaxing = True
        result = super().sample(x_dict, sample_shape, return_all, reparam, **kwargs)
        self.relaxing = tmp
        return result

    def sample_mean(self, x_dict=None):
        tmp = self.relaxing
        self.relaxing = False
        result = super().sample_mean(x_dict)
        self.relaxing = tmp
        return result

    def sample_variance(self, x_dict=None):
        tmp = self.relaxing
        self.relaxing = False
        result = super().sample_variance(x_dict)
        self.relaxing = tmp
        return result


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

    def get_log_prob(self, x_dict, **kwargs):
        x_dict = SampleDict.from_arg(x_dict, required_keys=self.var + self._cond_var)
        _x_dict = x_dict.from_variables(self._cond_var)
        self.set_dist(_x_dict)

        x_target = x_dict[self.var[0]]
        log_prob = self.dist.log_prob(x_target)
        log_prob[x_target == 0] = 0
        # sum over a factorized dim
        start, end = x_dict.features_dims(self.var[0])
        end = min(end, log_prob.ndim)
        dim = list(range(start, end))
        if dim:
            log_prob = log_prob.sum(dim=dim)
        return log_prob


class Categorical(DistributionBase):
    """Categorical distribution parameterized by :attr:`probs`."""

    @property
    def params_keys(self):
        return ["probs", "logits"]

    @property
    def distribution_torch_class(self):
        return CategoricalTorch

    @property
    def distribution_name(self):
        return "Categorical"


class RelaxedCategorical(Categorical):
    """Relaxed (re-parameterizable) categorical distribution parameterized by :attr:`probs`."""

    def __init__(self, temperature=torch.tensor(0.1), cond_var=(), var=("x",), name="p", features_shape=torch.Size(),
                 **kwargs):
        self._temperature = temperature
        # default is False for KL divergence
        self.relaxing = False

        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return CategoricalTorch if not self.relaxing else RelaxedOneHotCategoricalTorch

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def set_dist(self, x_dict=None, **dist_options):
        if self.relaxing:
            super().set_dist(x_dict, temperature=self.temperature, **dist_options)
        else:
            super().set_dist(x_dict, **dist_options)

    def sample(self, x_dict=None, sample_shape=torch.Size(), return_all=True, reparam=False, **kwargs):
        tmp = self.relaxing
        self.relaxing = True
        result = super().sample(x_dict, sample_shape, return_all, reparam, **kwargs)
        self.relaxing = tmp
        return result

    def sample_mean(self, x_dict=None):
        tmp = self.relaxing
        self.relaxing = False
        result = super().sample_mean(x_dict)
        self.relaxing = tmp
        return result

    def sample_variance(self, x_dict=None):
        tmp = self.relaxing
        self.relaxing = False
        result = super().sample_variance(x_dict)
        self.relaxing = tmp
        return result


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
        return ["probs", "logits", "total_count"]

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
