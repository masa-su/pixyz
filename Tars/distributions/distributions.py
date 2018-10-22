from __future__ import print_function
import torch
from torch import nn
from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import RelaxedBernoulli as RelaxedBernoulliTorch
from torch.distributions \
    import RelaxedOneHotCategorical as RelaxedOneHotCategoricalTorch
from torch.distributions.one_hot_categorical\
    import OneHotCategorical as CategoricalTorch

from ..utils import get_dict_values
from .operators import MultiplyDistribution


class Distribution(nn.Module):
    """
    Distribution class. In Tars, all distributions are required to inherit this class.

    Attributes
    ----------
    var : list
        Variables of this distribution.

    cond_var : list
        Conditional variables of this distribution.
        In case that cond_var is not empty, we must set the corresponding inputs in order to
        sample variables or estimate the log likelihood.

    dim : int
        Number of dimensions of this distribution.
        This might be ignored depending on the shape which is set in the sample method and on its parent distribution.
        Moreover, this is not consider when this class is inherited by DNNs.
        This is set to 1 by default.

    name : str
        Name of this distribution.
        This name is displayed in prob_text and prob_factorized_text.
        This is set to "p" by default.
    """

    def __init__(self, cond_var=[], var=["x"], name="p", dim=1,
                 **kwargs):
        super().__init__()
        self.cond_var = cond_var
        self.var = var
        self.dim = dim
        self.name = name

        # these members are intended to be overrided.
        # self.dist = None
        # self.distribution_name = None
        # self.params_keys = None

        self._initialize_constant_params(**kwargs)
        self._update_prob_text()

    def _check_input(self, x, var=None):
        """
        Check the type of a given input.
        If this type is a dictionary, we check whether this key and `var` are same.
        In case that this is list or tensor, we return a output formatted in a dictionary.

        Parameters
        ----------
        x : torch.Tensor, list, or dict
            Input variables

        var : list or None
            Variables to check if `x` has them.
            This is set to None by default.

        Returns
        -------
        checked_x : dict
            Variables which are checked in this method.

        Raises
        ------
        ValueError
            Raises ValueError if the type of `x` is neither tensor, list, nor dictionary.
        """

        if var is None:
            var = self.cond_var

        if type(x) is torch.Tensor:
            checked_x = {var[0]: x}

        elif type(x) is list:
            checked_x = dict(zip(var, x))

        elif type(x) is dict:
            if not set(list(x.keys())) == set(var):
                raise ValueError("Input's keys are not valid.")
            checked_x = x

        else:
            raise ValueError("The type of input is not valid, got %s."
                             % type(x))

        return checked_x

    def _initialize_constant_params(self, **params):
        """
        Format constant parameters set at initialization of this distribution.

        Parameters
        ----------
        params : dict
            Constant parameters of this distribution set at initialization.
            If the values of these dictionaries contain parameters which are named as strings, which means that
            these parameters are set as "variables", the correspondences between these values and the true name of
            these parameters are stored as a dictionary format (map_dict).
        """

        self.constant_params = {}
        self.map_dict = {}

        for keys in self.params_keys:
            if keys in params.keys():
                if type(params[keys]) is str:
                    self.map_dict[params[keys]] = keys
                else:
                    self.constant_params[keys] = params[keys]

        # Set a distribution if all parameters are constant and
        # set at initialization.
        if len(self.constant_params) == len(self.params_keys):
            self._set_distribution()

    def _update_prob_text(self):
        """
        Update `prob_text` from `cond_var`, `var`, and `name`.
        """

        _prob_text = [','.join(self.var)]
        if len(self.cond_var) != 0:
            _prob_text += [','.join(self.cond_var)]

        self.prob_text = "{}({})".format(
            self.name,
            "|".join(_prob_text)
        )

        self._update_prob_factorized_text()

    def _update_prob_factorized_text(self):
        """
        Update `prob_factorized_text`.
        Because this class models a single distribution, this is same as `prob_text`.
        """

        self.prob_factorized_text = self.prob_text

    def _set_distribution(self, x={}, **kwargs):
        params = self.get_params(x, **kwargs)
        self.dist = self.DistributionTorch(**params)

    def _get_sample(self, reparam=True,
                    sample_shape=torch.Size()):
        """
        Parameters
        ----------
        reparam : bool

        sample_shape : tuple

        Returns
        -------
        samples_dict : dict

        """

        if reparam:
            try:
                _samples = self.dist.rsample(sample_shape=sample_shape)
            except NotImplementedError:
                print("We can not use the reparameterization trick"
                      "for this distribution.")
        else:
            _samples = self.dist.sample(sample_shape=sample_shape)
        samples_dict = {self.var[0]: _samples}

        return samples_dict

    def _get_log_like(self, x):
        """
        Parameters
        ----------
        x : dict

        Returns
        -------
        log_like : torch.Tensor

        """

        x_targets = get_dict_values(x, self.var)
        log_like = self.dist.log_prob(*x_targets)

        return log_like

    def _map_variables_to_params(self, **variables):
        """
        Replace variables in keys of a input dictionary to parameters of this distribution according to
        these correspondences which is formatted in a dictionary and set in `_initialize_constant_params`.

        Parameters
        ----------
        variables : dict

        Returns
        -------
        mapped_params : dict

        variables : dict

        Examples
        --------
        >> distribution.map_dict
        >> > {"a": "loc"}
        >> x = {"a": 0}
        >> distribution._map_variables_to_params(x)
        >> > {"loc": 0}, {}
        """

        mapped_params = {self.map_dict[key]: value for key, value in variables.items()
                         if key in list(self.map_dict.keys())}

        variables = {key: value for key, value in variables.items()
                     if key not in list(self.map_dict.keys())}

        return mapped_params, variables

    def set_name(self, name):
        self.name = name
        self._update_prob_text()

    def get_params(self, params):
        """
        This method aims to get parameters of this distributions from constant parameters set in
        initialization and outputs of DNNs.

        Parameters
        ----------
        params : dict

        Returns
        -------
        output : dict

        Examples
        --------
        >> print(dist_1.prob_text, dist_1.distribution_name)
        >> > p(x) Normal
        >> dist_1.get_params()
        >> > {"loc": 0, "scale": 1}
        >> print(dist_2.prob_text, dist_2.distribution_name)
        >> > p(x|z) Normal
        >> dist_1.get_params({"z": 1})
        >> > {"loc": 0, "scale": 1}
        """

        params, variables = self._map_variables_to_params(**params)
        output = self.forward(**variables)

        # append constant_params to dict
        output.update(params)
        output.update(self.constant_params)

        return output

    def sample(self, x=None, shape=None, batch_size=1, return_all=True,
               reparam=True, **kwargs):
        """
        Sample variables of this distribution.
        If `cond_var` is not empty, we should set inputs as a dictionary format.

        Parameters
        ----------
        x : torch.Tensor, list, or dict
            Input variables.

        shape : tuple
            Shape of samples.
            If set, `batch_size` and `dim` are ignored.

        batch_size : int
            Batch size of samples. This is set to 1 by default.

        return_all : bool
            Choose whether the output contains input variables.

        reparam : bool
            Choose whether we sample variables with reparameterized trick.

        kwargs : dict

        Returns
        -------
        output : dict
            Samples of this distribution.
        """

        if x is None:  # unconditioned
            if len(self.cond_var) != 0:
                raise ValueError("You should set inputs or parameters")

            if shape:
                sample_shape = shape
            else:
                sample_shape = (batch_size, self.dim)

            output = self._get_sample(reparam=reparam,
                                      sample_shape=sample_shape)

        else:  # conditioned
            x = self._check_input(x)
            self._set_distribution(x, **kwargs)
            output = self._get_sample(reparam=reparam)

            if return_all:
                output.update(x)

        return output

    def log_likelihood(self, x):
        """
        Estimate the log likelihood of this distribution from inputs formatted by a dictionary.

        Parameters
        ----------
        x : dict


        Returns
        -------
        log_like : torch.Tensor

        """

        if not set(list(x.keys())) >= set(self.cond_var + self.var):
            raise ValueError("Input's keys are not valid.")

        if len(self.cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self.cond_var, True)
            self._set_distribution(_x)

        log_like = self._get_log_like(x)
        log_like = mean_sum_samples(log_like)
        return log_like

    def forward(self, **params):
        """
        When this class is inherited by DNNs, it is also intended that this method is overrided.

        Parameters
        ----------
        params : dict


        Returns
        -------
        params : dict

        """

        return params

    def sample_mean(self):
        NotImplementedError

    def __mul__(self, other):
        return MultiplyDistribution(self, other)

    def __str__(self):
        return self.prob_text


class CustomLikelihoodDistribution(Distribution):

    def __init__(self, var=["x"],  likelihood=None,
                 **kwargs):
        if likelihood is None:
            raise ValueError("You should set the likelihood"
                             " of this distribution.")
        self.likelihood = likelihood
        self.params_keys = []
        self.distribution_name = "Custom Distribution"
        self.DistributionTorch = None

        super().__init__(var=var, cond_var=[], **kwargs)

    def _set_distribution(self, x={}):
        pass

    def _get_log_like(self, x):
        # input : dict
        # output : tensor

        x_targets = get_dict_values(x, self.var)
        return torch.log(self.likelihood(x_targets[0]))

    def get_params(self, **kwargs):
        pass

    def sample(self, **kwargs):
        pass


class Normal(Distribution):

    def __init__(self, **kwargs):
        self.params_keys = ["loc", "scale"]
        self.distribution_name = "Normal"
        self.DistributionTorch = NormalTorch

        super().__init__(**kwargs)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["loc"]


class Bernoulli(Distribution):

    def __init__(self, *args, **kwargs):
        self.params_keys = ["probs"]
        self.distribution_name = "Bernoulli"
        self.DistributionTorch = BernoulliTorch

        super().__init__(*args, **kwargs)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class RelaxedBernoulli(Distribution):

    def __init__(self, temperature,
                 *args, **kwargs):
        self.params_keys = ["probs"]
        self.distribution_name = "RelaxedBernoulli"
        self.DistributionTorch = BernoulliTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedBernoulliTorch
        self.temperature = temperature

        super().__init__(*args, **kwargs)

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

        if not set(list(x.keys())) >= set(self.cond_var + self.var):
            raise ValueError("Input's keys are not valid.")

        if len(self.cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self.cond_var, True)
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

    def _get_log_like(self, x):
        log_like = super()._get_log_like(x)
        [_x] = get_dict_values(x, self.var)
        log_like[_x == 0] = 0
        return log_like


class Categorical(Distribution):

    def __init__(self, one_hot=True, *args, **kwargs):
        self.one_hot = one_hot
        self.params_keys = ["probs"]
        self.distribution_name = "Categorical"
        self.DistributionTorch = CategoricalTorch

        super().__init__(*args, **kwargs)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


class RelaxedCategorical(Distribution):

    def __init__(self, temperature,
                 *args, **kwargs):
        self.params_keys = ["probs"]
        self.distribution_name = "RelaxedCategorical"
        self.DistributionTorch = CategoricalTorch
        # use relaxed version only when sampling
        self.RelaxedDistributionTorch = RelaxedOneHotCategoricalTorch
        self.temperature = temperature

        super().__init__(*args, **kwargs)

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

        if not set(list(x.keys())) >= set(self.cond_var + self.var):
            raise ValueError("Input's keys are not valid.")

        if len(self.cond_var) > 0:  # conditional distribution
            _x = get_dict_values(x, self.cond_var, True)
            self._set_distribution(_x, sampling=False)

        log_like = self._get_log_like(x)
        return mean_sum_samples(log_like)

    def sample_mean(self, x):
        params = self.forward(**x)
        return params["probs"]


def mean_sum_samples(samples):
    dim = samples.dim()
    if dim == 4:
        return torch.mean(torch.sum(torch.sum(samples, dim=2), dim=2), dim=1)
    elif dim == 3:
        return torch.sum(torch.sum(samples, dim=-1), dim=-1)
    elif dim == 2:
        return torch.sum(samples, dim=-1)
    elif dim == 1:
        return samples
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % dim)
