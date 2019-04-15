from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import RelaxedBernoulli as RelaxedBernoulliTorch
from torch.distributions \
    import RelaxedOneHotCategorical as RelaxedOneHotCategoricalTorch
from torch.distributions.one_hot_categorical\
    import OneHotCategorical as CategoricalTorch
from torch.distributions import Multinomial as MultinomialTorch
from torch.distributions import Dirichlet as DirichletTorch
from torch.distributions import Beta as BetaTorch
from torch.distributions import Laplace as LaplaceTorch
from torch.distributions import Gamma as GammaTorch

from ..utils import get_dict_values
from .distributions import DistributionBase, sum_samples


class Normal(DistributionBase):
    """
    Normal distribution parameterized by :attr:`loc` and :attr:`scale`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["loc", "scale"]
        self.DistributionTorch = NormalTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Normal"


class Bernoulli(DistributionBase):
    """
    Bernoulli distribution parameterized by :attr:`probs`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = BernoulliTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Bernoulli"


class RelaxedBernoulli(DistributionBase):
    """
    Relaxed (reparameterizable) Bernoulli distribution parameterized by :attr:`probs`.
    """

    def __init__(self, temperature, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = BernoulliTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedBernoulliTorch
        self.temperature = temperature

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def set_distribution(self, x={}, sampling=True, **kwargs):
        params = self.get_params(x, **kwargs)
        if sampling is True:
            self.dist =\
                self.RelaxedDistributionTorch(temperature=self.temperature,
                                              **params)
        else:
            self.dist = self.DistributionTorch(**params)

    def log_likelihood(self, x):
        # input : dict
        # output : dict

        if not set(list(x.keys())) >= set(self._cond_var + self._var):
            raise ValueError("Input keys are not valid.")

        if len(self._cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self._cond_var, True)
            self.set_distribution(_x, sampling=False)

        log_like = self._get_log_like(x)
        return sum_samples(log_like)


class FactorizedBernoulli(Bernoulli):
    """
    Factorized Bernoulli distribution parameterized by :attr:`probs`.

    See `Generative Models of Visually Grounded Imagination`
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "FactorizedBernoulli"

    def _get_log_like(self, x):
        log_like = super()._get_log_like(x)
        [_x] = get_dict_values(x, self._var)
        log_like[_x == 0] = 0
        return log_like


class Categorical(DistributionBase):
    """
    Categorical distribution parameterized by :attr:`probs`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = CategoricalTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Categorical"


class RelaxedCategorical(DistributionBase):
    """
    Relaxed (reparameterizable) categorical distribution parameterized by :attr:`probs`.
    """

    def __init__(self, temperature, cond_var=[], var=["x"], name="p", dim=None,
                 **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = CategoricalTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedOneHotCategoricalTorch
        self.temperature = temperature

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def set_distribution(self, x={}, sampling=True, **kwargs):
        params = self.get_params(x, **kwargs)
        if sampling is True:
            self.dist =\
                self.RelaxedDistributionTorch(temperature=self.temperature,
                                              **params)
        else:
            self.dist = self.DistributionTorch(**params)

    def log_likelihood(self, x):
        # input : dict
        # output : dict

        if not set(list(x.keys())) >= set(self._cond_var + self._var):
            raise ValueError("Input keys are not valid.")

        if len(self._cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self._cond_var, True)
            self.set_distribution(_x, sampling=False)

        log_like = self._get_log_like(x)
        return sum_samples(log_like)


class Multinomial(DistributionBase):
    """
    Multinomial distribution parameterized by :attr:`total_count` and :attr:`probs`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["total_count", "probs"]
        self.DistributionTorch = MultinomialTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Multinomial"


class Dirichlet(DistributionBase):
    """
    Dirichlet distribution parameterized by :attr:`concentration`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["concentration"]
        self.DistributionTorch = DirichletTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Dirichlet"


class Beta(DistributionBase):
    """
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["concentration1", "concentration0"]
        self.DistributionTorch = BetaTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Beta"


class Laplace(DistributionBase):
    """
    Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["loc", "scale"]
        self.DistributionTorch = LaplaceTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Laplace"


class Gamma(DistributionBase):
    """
    Gamma distribution parameterized by :attr:`concentration` and :attr:`rate`.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=None, **kwargs):
        self.params_keys = ["concentration", "rate"]
        self.DistributionTorch = GammaTorch

        super().__init__(cond_var=cond_var, var=var, name=name, dim=dim, **kwargs)

    @property
    def distribution_name(self):
        return "Gamma"
