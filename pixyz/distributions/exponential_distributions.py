from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import RelaxedBernoulli as RelaxedBernoulliTorch
from torch.distributions \
    import RelaxedOneHotCategorical as RelaxedOneHotCategoricalTorch
from torch.distributions.one_hot_categorical\
    import OneHotCategorical as CategoricalTorch

from ..utils import get_dict_values
from .distributions import DistributionBase, mean_sum_samples


class Normal(DistributionBase):

    def __init__(self, **kwargs):
        self.params_keys = ["loc", "scale"]
        self.DistributionTorch = NormalTorch

        super().__init__(**kwargs)

    @property
    def distribution_name(self):
        return "Normal"

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["loc"]


class Bernoulli(DistributionBase):

    def __init__(self, *args, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = BernoulliTorch

        super().__init__(*args, **kwargs)

    @property
    def distribution_name(self):
        return "Bernoulli"

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class RelaxedBernoulli(DistributionBase):

    def __init__(self, temperature,
                 *args, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = BernoulliTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedBernoulliTorch
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def _set_distribution(self, x={}, sampling=True, **kwargs):
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
            raise ValueError("Input's keys are not valid.")

        if len(self._cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self._cond_var, True)
            self._set_distribution(_x, sampling=False)

        log_like = self._get_log_like(x)
        return mean_sum_samples(log_like)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class FactorizedBernoulli(Bernoulli):
    """
    Generative Models of Visually Grounded Imagination
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def distribution_name(self):
        return "FactorizedBernoulli"

    def _get_log_like(self, x):
        log_like = super()._get_log_like(x)
        [_x] = get_dict_values(x, self._var)
        log_like[_x == 0] = 0
        return log_like


class Categorical(DistributionBase):

    def __init__(self, one_hot=True, *args, **kwargs):
        self.one_hot = one_hot
        self.params_keys = ["probs"]
        self.DistributionTorch = CategoricalTorch

        super().__init__(*args, **kwargs)

    @property
    def distribution_name(self):
        return "Categorical"

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class RelaxedCategorical(DistributionBase):

    def __init__(self, temperature,
                 *args, **kwargs):
        self.params_keys = ["probs"]
        self.DistributionTorch = CategoricalTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedOneHotCategoricalTorch
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def _set_distribution(self, x={}, sampling=True, **kwargs):
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
            raise ValueError("Input's keys are not valid.")

        if len(self._cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self._cond_var, True)
            self._set_distribution(_x, sampling=False)

        log_like = self._get_log_like(x)
        return mean_sum_samples(log_like)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]
