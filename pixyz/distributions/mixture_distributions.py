import torch
from torch import nn

from ..distributions.distributions import Distribution


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
    >>> print(p.prob_text)
    p(x)
    >>> print(p.prob_factorized_text)
    p_0(x|z=0)prior(z=0) + p_1(x|z=1)prior(z=1) + p_2(x|z=2)prior(z=2)
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
            raise ValueError
        else:
            distributions = nn.ModuleList(distributions)

        if prior.distribution_name != "Categorical":
            raise ValueError("The prior must be the categorical distribution.")

        # check the number of mixture
        if len(prior.get_params()["probs"]) != len(distributions):
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

        self._distributions = distributions
        self._prior = prior

        self._hidden_var = hidden_var

    @property
    def prob_text(self):
        _prob_text = "{}({})".format(
            self._name, ','.join(self._var)
        )

        return _prob_text

    @property
    def prob_factorized_text(self):
        _mixture_prob_text = []
        for i, d in enumerate(self._distributions):
            _mixture_prob_text.append("{}({}|{}={}){}({}={})".format(
                d.name, self._var[0], self._hidden_var[0], i,
                self._prior.name, self._hidden_var[0], i
            ))

        _prob_text = ' + '.join(_mixture_prob_text)

        return _prob_text

    @property
    def distribution_name(self):
        return "Mixture Model"

    def posterior(self, name=None):
        return PosteriorMixtureModel(self, name=name)

    def sample(self, batch_size=1, return_hidden=False, **kwargs):
        hidden_output = []
        var_output = []

        for i in range(batch_size):
            # sample from prior
            _hidden_output = self._prior.sample()[self._hidden_var[0]]
            hidden_output.append(_hidden_output)

            var_output.append(self._distributions[_hidden_output.argmax(dim=-1)].sample()[self._var[0]])

        output_dict = {self._var[0]: torch.cat(var_output, 0)}

        if return_hidden:
            output_dict.update({self._hidden_var[0]: torch.cat(hidden_output, 0)})

        return output_dict

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
        eye_tensor = torch.eye(len(self._distributions)).to(_device)  # for prior

        for i, d in enumerate(self._distributions):
            # p(z=i)
            prior_log_prob = self._prior.log_prob().eval({self._hidden_var[0]: eye_tensor[i]})
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

        self._p = p
        self._hidden_var = p._hidden_var

    @property
    def prob_text(self):
        _prob_text = "{}({}|{})".format(
            self._name, self._hidden_var[0], self._var[0]
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

    def sample(self, x={}, shape=None, batch_size=1, return_all=True, reparam=False):
        raise NotImplementedError

    def get_log_prob(self, x_dict, **kwargs):
        # log p(z|x) = log p(x, z) - log p(x)
        log_prob = self._p.get_log_prob(x_dict, return_hidden=True) - self._p.get_log_prob(x_dict)
        return log_prob  # (num_mix, batch_size)
