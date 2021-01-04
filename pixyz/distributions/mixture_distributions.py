import torch
from torch import nn

from ..distributions.distributions import Distribution
from ..utils import convert_latex_name


class MixtureModel(Distribution):
    r"""Mixture models.

    .. math::

        p(x) = \sum_i p(x|z=i)p(z=i)

    Examples
    --------
    >>> from pixyz.distributions import Normal, Categorical
    >>> from pixyz.distributions.mixture_distributions import MixtureModel
    >>> z_dim = 3  # the number of mixture
    >>> x_dim = 2  # the input dimension.
    >>> distributions = []  # the list of distributions
    >>> for i in range(z_dim):
    ...     loc = torch.randn(x_dim)  # initialize the value of location (mean)
    ...     scale = torch.empty(x_dim).fill_(1.)  # initialize the value of scale (variance)
    ...     distributions.append(Normal(loc=loc, scale=scale, var=["x"], name="p_%d" %i))
    >>> probs = torch.empty(z_dim).fill_(1. / z_dim)  # initialize the value of probabilities
    >>> prior = Categorical(probs=probs, var=["z"], name="prior")
    >>> p = MixtureModel(distributions=distributions, prior=prior)
    >>> print(p)
    Distribution:
      p(x) = p_{0}(x|z=0)prior(z=0) + p_{1}(x|z=1)prior(z=1) + p_{2}(x|z=2)prior(z=2)
    Network architecture:
      MixtureModel(
        name=p, distribution_name=Mixture Model,
        var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([])
        (distributions): ModuleList(
          (0): Normal(
            name=p_{0}, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([2])
            (loc): torch.Size([1, 2])
            (scale): torch.Size([1, 2])
          )
          (1): Normal(
            name=p_{1}, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([2])
            (loc): torch.Size([1, 2])
            (scale): torch.Size([1, 2])
          )
          (2): Normal(
            name=p_{2}, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([2])
            (loc): torch.Size([1, 2])
            (scale): torch.Size([1, 2])
          )
        )
        (prior): Categorical(
          name=prior, distribution_name=Categorical,
          var=['z'], cond_var=[], input_var=[], features_shape=torch.Size([3])
          (probs): torch.Size([1, 3])
        )
      )
    """

    def __init__(self, distributions, prior, name="p"):
        """
        Parameters
        ----------
        distributions : list
            List of distributions.
        prior : pixyz.Distribution.Categorical
            Prior distribution of latent variable (i.e., a contribution rate).
            This should be a categorical distribution and
            the number of its category should be the same as the length of :attr:`distributions`.
        name : :obj:`str`, defaults to "p"
            Name of this distribution.
            This name is displayed in :attr:`prob_text` and :attr:`prob_factorized_text`.

        """
        if not isinstance(distributions, list):
            raise ValueError()
        else:
            distributions = nn.ModuleList(distributions)

        if prior.distribution_name != "Categorical":
            raise ValueError("The prior must be the categorical distribution.")

        # check the number of mixture
        if prior.get_params()["probs"].shape[-1] != len(distributions):
            raise ValueError("The number of its category must be the same as the length of the distribution list.")

        # check whether all distributions have the same variable.
        var_list = []
        for d in distributions:
            var_list += d.var
        var_list = list(set(var_list))

        if len(var_list) != 1:
            raise ValueError("All distributions must have the same variable.")

        hidden_var = prior.var

        super().__init__(var=var_list, name=name)

        self.distributions = distributions
        self.prior = prior

        self._hidden_var = hidden_var

    @property
    def hidden_var(self):
        """list: Hidden variables of this distribution."""
        return self._hidden_var

    @property
    def prob_factorized_text(self):
        _mixture_prob_text = []
        for i, d in enumerate(self.distributions):
            _mixture_prob_text.append("{}({}|{}={}){}({}={})".format(
                d.name, self.var[0], self._hidden_var[0], i,
                self.prior.name, self._hidden_var[0], i
            ))

        _prob_text = ' + '.join(_mixture_prob_text)

        return _prob_text

    @property
    def distribution_name(self):
        return "Mixture Model"

    def posterior(self, name=None):
        return PosteriorMixtureModel(self, name=name)

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, return_hidden=False,
               bypass_from=None, **kwargs):
        """Sample variables of this distribution.
        If :attr:`cond_var` is not empty, you need to set :attr:`x_dict` containing values of :attr:`cond_var`.

        Parameters
        ----------
        x_dict : :obj:`torch.Tensor`, :obj:`list`, or :obj:`dict`, defaults to {}
            Input variables.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        sample_shape : :obj:`list` or :obj:`NoneType`, defaults to torch.Size()
            Shape of generating samples.
        return_all : :obj:`bool`, defaults to True
            Choose whether the output contains input variables.
        return_hidden : :obj: `bool`, defaults to False
            Choose whether the output contains the hidden variable.
        bypass_from : :obj:`str`, defaults to None.
            Choose a parameter name to bypass stochastic sampling. The value of parameter is returned as a sample.

        Returns
        -------
        output : dict
            Samples of this distribution.

        Examples
        --------
        >>> from pixyz.distributions import Normal, Categorical
        >>> dist = MixtureModel([Normal(loc=0, scale=1), Normal(loc=1, scale=1)],
        ...                     Categorical(probs=torch.tensor([0., 1.])))
        >>> print(dist.sample()) # doctest: +SKIP
        {'x': tensor([0.6070])}
        >>> print(dist.sample(bypass_from="loc"))
        {'x': tensor([1.])}
        """
        # sample from prior
        hidden_output = self.prior.sample(batch_n=batch_n, sample_shape=sample_shape)[self._hidden_var[0]]

        var_output = []
        if sample_shape != torch.Size():
            # MixtureModel does not support sample_shape option because for loop is not scalable.
            raise ValueError("sample_shape option is not supported for MixtureModel.")
        for _hidden_output in hidden_output:
            dist = self.distributions[_hidden_output.argmax(dim=-1)]
            sample = dist.sample(bypass_from=bypass_from)[self._var[0]]
            var_output.append(sample)

        var_output = torch.cat(var_output, dim=0)
        output_dict = {self._var[0]: var_output}

        if return_hidden:
            output_dict.update({self._hidden_var[0]: hidden_output})

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    @property
    def has_reparam(self):
        return False

    def get_log_prob(self, x_dict, return_hidden=False, **kwargs):
        """Evaluate log-pdf, log p(x) (if return_hidden=False) or log p(x, z) (if return_hidden=True).

        Parameters
        ----------
        x_dict : dict
            Input variables (including `var`).

        return_hidden : :obj:`bool`, defaults to False

        Returns
        -------
        log_prob : torch.Tensor
            The log-pdf value of x.

            return_hidden = 0 :
                dim=0 : the size of batch

            return_hidden = 1 :
                dim=0 : the number of mixture

                dim=1 : the size of batch

        """

        log_prob_all = []

        _device = x_dict[self._var[0]].device
        eye_tensor = torch.eye(len(self.distributions)).to(_device)  # for prior

        for i, d in enumerate(self.distributions):
            # p(z=i)
            prior_log_prob = self.prior.log_prob().eval({self._hidden_var[0]: eye_tensor[i]})
            # p(x|z=i)
            log_prob = d.log_prob().eval(x_dict)
            # p(x, z=i)
            log_prob_all.append(log_prob + prior_log_prob)

        log_prob_all = torch.stack(log_prob_all, dim=0)  # (num_mix, batch_size)

        if return_hidden:
            return log_prob_all

        return torch.logsumexp(log_prob_all, 0)


class PosteriorMixtureModel(Distribution):
    def __init__(self, p, name=None):
        if name is None:
            name = p.name
        super().__init__(var=p.var, name=name)

        self.p = p
        self._hidden_var = p.hidden_var

    @property
    def hidden_var(self):
        """list: Hidden variables of this distribution."""
        return self._hidden_var

    @property
    def prob_text(self):
        _prob_text = "{}({}|{})".format(
            self._name, convert_latex_name(self._hidden_var[0]), convert_latex_name(self._var[0])
        )
        return _prob_text

    @property
    def prob_factorized_text(self):
        numinator = "{" + "{}({},{})".format(self._name, self._hidden_var[0], self._var[0]) + "}"
        denominator = "{" + "{}({})".format(self._name, self._var[0]) + "}"

        _prob_text = "\\frac{}{}".format(numinator, denominator)

        return _prob_text

    @property
    def distribution_name(self):
        return "Mixture Model (Posterior)"

    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def has_reparam(self):
        return False

    def get_log_prob(self, x_dict, **kwargs):
        # log p(z|x) = log p(x, z) - log p(x)
        log_prob = self.p.get_log_prob(x_dict, return_hidden=True) - self.p.get_log_prob(x_dict)
        return log_prob  # (num_mix, batch_size)
