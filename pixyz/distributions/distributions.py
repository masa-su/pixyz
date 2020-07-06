from __future__ import print_function
import torch
import re
import networkx as nx
from torch import nn

from ..utils import get_dict_values, replace_dict_keys_split, delete_dict_values,\
    tolist, sum_samples, convert_latex_name, lru_cache_for_sample_dict
from ..losses import LogProb, Prob


class DistGraph:
    def __init__(self, original=None, name='p'):
        self.graph = nx.DiGraph()
        self.global_option = {}
        self.marginalize_list = set()
        self.name = convert_latex_name(name)
        if original:
            self.graph.update(original.graph)
            self.global_option.update(original.global_option)
            self.marginalize_list.update(original.marginalize_list)

    def appended(self, atom_dist, var, cond_var=(), name=''):
        if not name:
            name = self.name
        scg = DistGraph(self, name=name)
        var = var[0]
        if var in scg.graph:
            raise ValueError()
        scg.graph.add_node(var, atom=atom_dist, name_dict={})
        for cond in cond_var:
            scg.graph.add_edge(cond, var)
        return scg

    def set_option(self, option_dict, var=[]):
        if not var:
            self.global_option.update(option_dict)
        else:
            self.graph.add_node(var, option=option_dict)

    def united(self, other):
        if not set(self.var + list(self.marginalize_list)).isdisjoint(set(other.var + list(other.marginalize_list))):
            raise ValueError()
        scg = DistGraph(self)
        scg.graph.update(other.graph)
        scg.marginalize_list.update(other.marginalize_list)
        return scg

    def marginalized(self, marginalize_list):
        if len(marginalize_list) == 0:
            raise ValueError("Length of `marginalize_list` must be at least 1, got 0.")
        if not set(marginalize_list) < set(self.var):
            raise ValueError("marginalize_list has unknown variables or it has all of variables of `p`.")
        if not ((set(marginalize_list)).isdisjoint(set(self.cond_var))):
            raise ValueError("Conditional variables can not be marginalized.")

        new_graph = DistGraph(self)
        new_graph.marginalize_list.update(marginalize_list)
        return new_graph

    def _replace_atom_name_dict(self, var, replace_dict):
        name_dict = self.graph.nodes(data='name_dict')[var]
        atom_var = list(self.graph.pred[var]) + [var]
        for avar in atom_var:
            if avar not in replace_dict:
                continue
            new_global_var = replace_dict[avar]
            if avar in name_dict:
                local_var = name_dict[avar]
                del name_dict[avar]
                name_dict[new_global_var] = local_var
            else:
                name_dict[new_global_var] = avar

    def var_replaced(self, replace_dict, name=''):
        if not (set(replace_dict.keys()) <= set(self.graph)):
            unknown_var = [var_name for var_name in replace_dict.keys() if var_name not in self.graph]
            raise ValueError("replace_dict has unknown variables: {}".format(unknown_var))
        if not set(replace_dict.values()).isdisjoint(set(self.graph)):
            used_var = [var_name for var_name in replace_dict.values() if var_name in self.graph]
            raise ValueError("{} is already used in this distribution.".format(used_var))
        if not name:
            name = self.name
        result = DistGraph(name=name)
        result.graph = nx.relabel_nodes(self.graph, replace_dict)
        result.marginalize_list = {replace_dict[var] for var in self.marginalize_list}
        result.global_option = dict(self.global_option)
        for var, dist in self.node_distributions():
            self._replace_atom_name_dict(var, replace_dict)
        return result

    def node_distribution(self, var):
        return self.graph.nodes(data='atom')[var]

    def node_option(self, var):
        option = self.graph.nodes(data='option')[var]
        return option if option else {}

    def node_distributions(self, sorted=False):
        vars = nx.topological_sort(self.graph) if sorted else self.graph
        for var in vars:
            dist = self.node_distribution(var)
            if dist:
                yield var, dist

    @property
    def input_var(self):
        return [var for var in self.graph
                if self.node_distribution(var) is None or isinstance(
                    self.node_distribution(var), type('DataDistribution'))]

    @property
    def cond_var(self):
        return [var for var in self.graph if self.node_distribution(var) is None]

    @property
    def var(self):
        return [var for var in self.graph
                if self.node_distribution(var) is not None and var not in self.marginalize_list]

    def _get_local_name(self, global_name, location):
        name_dict = self.graph.nodes(data='name_dict')[location]
        return global_name if global_name not in name_dict else name_dict[global_name]

    def _get_local_input_dict(self, values, var, atom_var):
        lvar = [self._get_local_name(var_name, atom_var) for var_name in var]
        return get_dict_values(values, lvar, return_dict=True)

    def _get_local_output_value(self, atom_var, output_dict):
        lvar = self._get_local_name(atom_var, atom_var)
        return output_dict[lvar]

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        sample_option = dict(self.global_option)
        sample_option.update(dict(batch_n=batch_n, sample_shape=sample_shape,
                                  return_all=False, reparam=reparam))
        # ignore return_all because overriding is now under control.
        if not(set(x_dict.keys()) >= set(self.input_var)):
            raise ValueError("Input keys are not valid, expected {} but got {}.".format(set(self.input_var),
                                                                                        set(x_dict.keys())))

        values = get_dict_values(x_dict, self.input_var, return_dict=True)
        for var, dist in self.node_distributions(sorted=True):
            if any(cond not in values for cond in self.graph.pred[var]):
                raise ValueError("lack of some condition variable")
            input_dict = self._get_local_input_dict(values, self.graph.pred[var], var)
            option = sample_option
            option.update(self.node_option(var))
            values[var] = self._get_local_output_value(var, dist.sample(input_dict, **option))
        result_dict = delete_dict_values(values, self.marginalize_list)
        if return_all:
            output_dict = dict(delete_dict_values(x_dict, self.input_var))
            output_dict.update(result_dict)
            return output_dict
        else:
            return delete_dict_values(result_dict, self.input_var)

    # def sample_scgraph(self):
    #     pass

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        if len(self.marginalize_list) > 0:
            # TODO: Deterministic分布のmarginalizeの場合はその変数が辞書に含まれない限りは対数尤度を定義してよい
            raise NotImplementedError()
        log_prob_option = dict(self.global_option)
        log_prob_option.update(dict(sum_features=sum_features, feature_dims=feature_dims))
        if not(set(x_dict.keys()) >= set(self.graph)):
            raise ValueError("Input keys are not valid, expected {} but got {}.".format(set(self.graph),
                                                                                        set(x_dict.keys())))

        values = get_dict_values(x_dict, list(self.graph), return_dict=True)
        log_prob = None
        prev_dist = None
        for var, dist in self.node_distributions(sorted=True):
            input_dict = self._get_local_input_dict(values, list(self.graph.pred[var]) + [var], var)
            option = log_prob_option
            option.update(self.node_option(var))
            new_log_prob = dist.get_log_prob(input_dict, **option)
            if log_prob is None:
                log_prob = new_log_prob
            else:
                if log_prob.size() != new_log_prob.size():
                    raise ValueError("Two PDFs, {} and {}, have different sizes,"
                                     " so you must modify these tensor sizes."
                                     .format(prev_dist.prob_text,
                                             dist.prob_text))
                log_prob += new_log_prob
            prev_dist = dist
        if log_prob is None:
            return 0
        return log_prob

    @property
    def has_reparam(self):
        return all(dist.has_reparam for _, dist in self.node_distributions())

    @property
    def prob_factorized_text(self):
        text = ""
        for var, dist in self.node_distributions(sorted=True):
            # factor_text = dist.prob_factorized_text
            factor_text = self.prob_node_text(var, dist)
            text = factor_text + text
        if self.marginalize_list:
            integral_symbol = len(self.marginalize_list) * "\\int "
            # TODO: var -> convert_latex_name(var) ?
            integral_variables = ["d" + var for var in self.marginalize_list]
            integral_variables = "".join(integral_variables)

            return "{}{}{}".format(integral_symbol, text, integral_variables)
        return text

    def _repr_atom(self, var, dist):
        prob_node_text = self.prob_node_text(var, dist)
        factorized_text = dist.prob_factorized_text
        header_text = f'{prob_node_text} =\n' if prob_node_text == factorized_text \
            else f'{prob_node_text} -> {factorized_text} =\n'
        return header_text + repr(dist)

    def __str__(self):
        # Distribution
        text = "Distribution:\n  {}\n".format(self.prob_joint_factorized_and_text)

        # Network architecture (`repr`)
        network_text = "\n".join(self._repr_atom(var, dist) for var, dist in self.node_distributions(sorted=True))
        network_text = re.sub('^', ' ' * 2, str(network_text), flags=re.MULTILINE)
        text += "Network architecture:\n{}".format(network_text)
        return text

    @property
    def prob_text(self):
        return "{}({}{})".format(self.name, ','.join(convert_latex_name(var_name) for var_name in self.var),
                                 '' if len(self.cond_var) == 0 else
                                 '|' + ','.join(convert_latex_name(var_name) for var_name in self.cond_var))

    def prob_node_text(self, var, dist):
        return "{}({}{})".format(dist.name, ','.join(convert_latex_name(var_name) for var_name in [var]),
                                 '' if len(self.graph.pred[var]) == 0 else
                                 '|' + ','.join(convert_latex_name(var_name) for var_name in self.graph.pred[var]))

    @property
    def prob_joint_factorized_and_text(self):
        if self.prob_factorized_text == self.prob_text:
            return self.prob_text
        else:
            return "{} = {}".format(self.prob_text, self.prob_factorized_text)


class Distribution(nn.Module):
    """Distribution class. In Pixyz, all distributions are required to inherit this class.


    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal
    >>> # Marginal distribution
    >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
    ...             features_shape=[64], name="p1")
    >>> print(p1)
    Distribution:
      p_{1}(x)
    Network architecture:
      p_{1}(x) =
      Normal(
        name=p_{1}, distribution_name=Normal,
        features_shape=torch.Size([64])
        (loc): torch.Size([1, 64])
        (scale): torch.Size([1, 64])
      )

    >>> # Conditional distribution
    >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
    ...             features_shape=[64], name="p2")
    >>> print(p2)
    Distribution:
      p_{2}(x|y)
    Network architecture:
      p_{2}(x|y) =
      Normal(
        name=p_{2}, distribution_name=Normal,
        features_shape=torch.Size([64])
        (scale): torch.Size([1, 64])
      )

    >>> # Conditional distribution (by neural networks)
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["x"], cond_var=["y"], name="p3")
    ...         self.model_loc = nn.Linear(128, 64)
    ...         self.model_scale = nn.Linear(128, 64)
    ...     def forward(self, y):
    ...         return {"loc": self.model_loc(y), "scale": F.softplus(self.model_scale(y))}
    >>> p3 = P()
    >>> print(p3)
    Distribution:
      p_{3}(x|y)
    Network architecture:
      p_{3}(x|y) =
      P(
        name=p_{3}, distribution_name=Normal,
        features_shape=torch.Size([])
        (model_loc): Linear(in_features=128, out_features=64, bias=True)
        (model_scale): Linear(in_features=128, out_features=64, bias=True)
      )
    """

    def __init__(self, var, cond_var=[], name="p", features_shape=torch.Size(), atomic=True):
        """
        Parameters
        ----------
        var : :obj:`list` of :obj:`str`
            Variables of this distribution.
        cond_var : :obj:`list` of :obj:`str`, defaults to []
            Conditional variables of this distribution.
            In case that cond_var is not empty, we must set the corresponding inputs to sample variables.
        name : :obj:`str`, defaults to "p"
            Name of this distribution.
            This name is displayed in :attr:`prob_text` and :attr:`prob_factorized_text`.
        features_shape : :obj:`torch.Size` or :obj:`list`, defaults to torch.Size())
            Shape of dimensions (features) of this distribution.

        """
        super().__init__()

        _vars = cond_var + var
        if len(_vars) != len(set(_vars)):
            raise ValueError("There are conflicted variables.")

        self._cond_var = cond_var
        self._var = var
        self._atomic = atomic
        if atomic:
            if len(var) != 1:
                raise ValueError("An atomic distribution must have only one variable.")
            self.graph = DistGraph().appended(atom_dist=self, var=var, cond_var=cond_var, name=name)
        else:
            self.graph = None

        self._features_shape = torch.Size(features_shape)
        self._name = convert_latex_name(name)

    @property
    def distribution_name(self):
        """str: Name of this distribution class."""
        return ""

    @property
    def name(self):
        """str: Name of this distribution displayed in :obj:`prob_text` and :obj:`prob_factorized_text`."""
        return self._name

    @name.setter
    def name(self, name):
        if type(name) is str:
            self._name = name
            self.graph.name = name
            return

        raise ValueError("Name of the distribution class must be a string type.")

    @property
    def var(self):
        """list: Variables of this distribution."""
        return self._var if self._atomic else self.graph.var

    @property
    def cond_var(self):
        """list: Conditional variables of this distribution."""
        return self._cond_var if self._atomic else self.graph.cond_var

    @property
    def input_var(self):
        """list: Input variables of this distribution.
        Normally, it has same values as :attr:`cond_var`.

        """
        return self._cond_var if self._atomic else self.graph.input_var

    @property
    def prob_text(self):
        """str: Return a formula of the (joint) probability distribution."""
        if not self._atomic:
            return self.graph.prob_text

        _var_text = [','.join([convert_latex_name(var_name) for var_name in self.var])]
        if len(self.cond_var) != 0:
            _var_text += [','.join([convert_latex_name(var_name) for var_name in self.cond_var])]

        _prob_text = "{}({})".format(
            self._name,
            "|".join(_var_text)
        )

        return _prob_text

    @property
    def prob_factorized_text(self):
        """str: Return a formula of the factorized probability distribution."""
        if not self._atomic:
            return self.graph.prob_factorized_text
        return self.prob_text

    @property
    def prob_joint_factorized_and_text(self):
        """str: Return a formula of the factorized and the (joint) probability distributions."""
        if not self._atomic:
            return self.graph.prob_joint_factorized_and_text

        if self.prob_factorized_text == self.prob_text:
            prob_text = self.prob_text
        else:
            prob_text = "{} = {}".format(self.prob_text, self.prob_factorized_text)
        return prob_text

    @property
    def features_shape(self):
        """torch.Size or list: Shape of features of this distribution."""
        return self._features_shape

    def _get_input_dict(self, input, var=None):
        """Check the type of given input.
        If the input type is :obj:`dict`, this method checks whether the input keys contains the :attr:`var` list.
        In case that its type is :obj:`list` or :obj:`tensor`, it returns the output formatted in :obj:`dict`.

        Parameters
        ----------
        input : :obj:`torch.Tensor`, :obj:`list`, or :obj:`dict`
            Input variables.
        var : :obj:`list` or :obj:`NoneType`, defaults to None
            Variables to check if given input contains them.
            This is set to None by default.

        Returns
        -------
        input_dict : dict
            Variables checked in this method.

        Raises
        ------
        ValueError
            Raises `ValueError` if the type of input is neither :obj:`torch.Tensor`, :obj:`list`, nor :obj:`dict.

        """
        if var is None:
            var = self.input_var

        if type(input) is torch.Tensor:
            input_dict = {var[0]: input}

        elif type(input) is list:
            # TODO: we need to check if all the elements contained in this list are torch.Tensor.
            input_dict = dict(zip(var, input))

        elif type(input) is dict:
            if not (set(list(input.keys())) >= set(var)):
                raise ValueError("Input keys are not valid.")
            input_dict = get_dict_values(input, var, return_dict=True)

        else:
            raise ValueError("The type of input is not valid, got %s." % type(input))

        return input_dict

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True,
               reparam=False):
        """Sample variables of this distribution.
        If :attr:`cond_var` is not empty, you should set inputs as :obj:`dict`.

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
        reparam : :obj:`bool`, defaults to False.
            Choose whether we sample variables with re-parameterized trick.

        Returns
        -------
        output : dict
            Samples of this distribution.

        Examples
        --------
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...            features_shape=[10, 2])
        >>> print(p)
        Distribution:
          p(x)
        Network architecture:
          p(x) =
          Normal(
            name=p, distribution_name=Normal,
            features_shape=torch.Size([10, 2])
            (loc): torch.Size([1, 10, 2])
            (scale): torch.Size([1, 10, 2])
          )
        >>> p.sample()["x"].shape  # (batch_n=1, features_shape)
        torch.Size([1, 10, 2])
        >>> p.sample(batch_n=20)["x"].shape  # (batch_n, features_shape)
        torch.Size([20, 10, 2])
        >>> p.sample(batch_n=20, sample_shape=[40, 30])["x"].shape  # (sample_shape, batch_n, features_shape)
        torch.Size([40, 30, 20, 10, 2])

        >>> # Conditional distribution
        >>> p = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...            features_shape=[10])
        >>> print(p)
        Distribution:
          p(x|y)
        Network architecture:
          p(x|y) =
          Normal(
            name=p, distribution_name=Normal,
            features_shape=torch.Size([10])
            (scale): torch.Size([1, 10])
          )
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> sample_a = torch.randn(1, 10) # Psuedo data
        >>> sample = p.sample({"y": sample_y})
        >>> print(sample) # input_var + var  # doctest: +SKIP
        {'y': tensor([[-0.5182,  0.3484,  0.9042,  0.1914,  0.6905,
                       -1.0859, -0.4433, -0.0255, 0.8198,  0.4571]]),
         'x': tensor([[-0.7205, -1.3996,  0.5528, -0.3059,  0.5384,
                       -1.4976, -0.1480,  0.0841,0.3321,  0.5561]])}
        >>> sample = p.sample({"y": sample_y, "a": sample_a}) # Redundant input ("a")
        >>> print(sample) # input_var + var + "a" (redundant input)  # doctest: +SKIP
        {'y': tensor([[ 1.3582, -1.1151, -0.8111,  1.0630,  1.1633,
                        0.3855,  2.6324, -0.9357, -0.8649, -0.6015]]),
         'a': tensor([[-0.1874,  1.7958, -1.4084, -2.5646,  1.0868,
                       -0.7523, -0.0852, -2.4222, -0.3914, -0.9755]]),
         'x': tensor([[-0.3272, -0.5222, -1.3659,  1.8386,  2.3204,
                        0.3686,  0.6311, -1.1208, 0.3656, -0.6683]])}

        """
        if self.graph:
            return self.graph.sample(x_dict, batch_n, sample_shape, return_all, reparam)
        raise NotImplementedError()

    @property
    def has_reparam(self):
        if self.graph:
            return self.graph.has_reparam
        raise NotImplementedError()

    def sample_mean(self, x_dict={}):
        """Return the mean of the distribution.

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}
            Parameters of this distribution.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> mean = p1.sample_mean()
        >>> print(mean)
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> mean = p2.sample_mean({"y": sample_y})
        >>> print(mean) # doctest: +SKIP
        tensor([[-0.2189, -1.0310, -0.1917, -0.3085,  1.5190, -0.9037,  1.2559,  0.1410,
                  1.2810, -0.6681]])

        """
        raise NotImplementedError()

    def sample_variance(self, x_dict={}):
        """Return the variance of the distribution.

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}
            Parameters of this distribution.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> var = p1.sample_variance()
        >>> print(var)
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> var = p2.sample_variance({"y": sample_y})
        >>> print(var) # doctest: +SKIP
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        """
        raise NotImplementedError()

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        """Giving variables, this method returns values of log-pdf.

        Parameters
        ----------
        x_dict : dict
            Input variables.
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some dimensions which are specified by `feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output.

        Returns
        -------
        log_prob : torch.Tensor
            Values of log-probability density/mass function.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p1.log_prob({"x": sample_x})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-16.1153])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p2.log_prob({"x": sample_x, "y": sample_y})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-21.5251])

        """
        if self.graph:
            return self.graph.get_log_prob(x_dict, sum_features, feature_dims)
        raise NotImplementedError()

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        """Giving variables, this method returns values of entropy.

        Parameters
        ----------
        x_dict : dict, defaults to {}
            Input variables.
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some dimensions which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output.

        Returns
        -------
        entropy : torch.Tensor
            Values of entropy.

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> entropy = p1.get_entropy()
        >>> print(entropy)
        tensor([14.1894])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> entropy = p2.get_entropy({"y": sample_y})
        >>> print(entropy)
        tensor([14.1894])

        """
        raise NotImplementedError()

    def log_prob(self, sum_features=True, feature_dims=None):
        """Return an instance of :class:`pixyz.losses.LogProb`.

        Parameters
        ----------
        sum_features : :obj:`bool`, defaults to True
            Whether the output is summed across some axes (dimensions) which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set axes to sum across the output.

        Returns
        -------
        pixyz.losses.LogProb
            An instance of :class:`pixyz.losses.LogProb`

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p1.log_prob().eval({"x": sample_x})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-16.1153])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> log_prob = p2.log_prob().eval({"x": sample_x, "y": sample_y})
        >>> print(log_prob) # doctest: +SKIP
        tensor([-21.5251])

        """
        return LogProb(self, sum_features=sum_features, feature_dims=feature_dims)

    def prob(self, sum_features=True, feature_dims=None):
        """Return an instance of :class:`pixyz.losses.LogProb`.

        Parameters
        ----------
        sum_features : :obj:`bool`, defaults to True
            Choose whether the output is summed across some axes (dimensions)
            which are specified by :attr:`feature_dims`.
        feature_dims : :obj:`list` or :obj:`NoneType`, defaults to None
            Set dimensions to sum across the output. (Note: this parameter is not used for now.)

        Returns
        -------
        pixyz.losses.Prob
            An instance of :class:`pixyz.losses.Prob`

        Examples
        --------
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> # Marginal distribution
        >>> p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...             features_shape=[10], name="p1")
        >>> sample_x = torch.randn(1, 10) # Psuedo data
        >>> prob = p1.prob().eval({"x": sample_x})
        >>> print(prob) # doctest: +SKIP
        tensor([4.0933e-07])

        >>> # Conditional distribution
        >>> p2 = Normal(loc="y", scale=torch.tensor(1.), var=["x"], cond_var=["y"],
        ...             features_shape=[10], name="p2")
        >>> sample_y = torch.randn(1, 10) # Psuedo data
        >>> prob = p2.prob().eval({"x": sample_x, "y": sample_y})
        >>> print(prob) # doctest: +SKIP
        tensor([2.9628e-09])

        """
        return Prob(self, sum_features=sum_features, feature_dims=feature_dims)

    def forward(self, *args, **kwargs):
        """When this class is inherited by DNNs, this method should be overrided."""

        raise NotImplementedError()

    def replace_var(self, **replace_dict):
        """Return an instance of :class:`pixyz.distributions.ReplaceVarDistribution`.

        Parameters
        ----------
        replace_dict : dict
            Dictionary.

        Returns
        -------
        pixyz.distributions.ReplaceVarDistribution
            An instance of :class:`pixyz.distributions.ReplaceVarDistribution`

        """

        return ReplaceVarDistribution(self, replace_dict)

    def marginalize_var(self, marginalize_list):
        """Return an instance of :class:`pixyz.distributions.MarginalizeVarDistribution`.

        Parameters
        ----------
        marginalize_list : :obj:`list` or other
            Variables to marginalize.

        Returns
        -------
        pixyz.distributions.MarginalizeVarDistribution
            An instance of :class:`pixyz.distributions.MarginalizeVarDistribution`

        """

        marginalize_list = tolist(marginalize_list)
        return MarginalizeVarDistribution(self, marginalize_list)

    def __mul__(self, other):
        return MultiplyDistribution(self, other)

    def __str__(self):
        if self.graph:
            return str(self.graph)
        # Distribution
        text = "Distribution:\n  {}\n".format(self.prob_joint_factorized_and_text)

        # Network architecture (`repr`)
        network_text = self.__repr__()
        network_text = re.sub('^', ' ' * 2, str(network_text), flags=re.MULTILINE)
        text += "Network architecture:\n{}".format(network_text)
        return text

    def extra_repr(self):
        # parameters
        parameters_text = 'name={}, distribution_name={},\n' \
                          'features_shape={}'.format(self.name, self.distribution_name,
                                                     self.features_shape
                                                     )

        if len(self._buffers) != 0:
            # add buffers to repr
            buffers = ["({}): {}".format(key, value.shape) for key, value in self._buffers.items()]
            return parameters_text + "\n" + "\n".join(buffers)

        return parameters_text


class DistributionBase(Distribution):
    """Distribution class with PyTorch. In Pixyz, all distributions are required to inherit this class."""

    def __init__(self, cond_var=[], var=["x"], name="p", features_shape=torch.Size(), **kwargs):
        super().__init__(cond_var=cond_var, var=var, name=name, features_shape=features_shape)

        self._set_buffers(**kwargs)
        self._dist = None

    def _set_buffers(self, **params_dict):
        """Format constant parameters of this distribution as buffers.

        Parameters
        ----------
        params_dict : dict
            Constant parameters of this distribution set at initialization.
            If the values of these dictionaries contain parameters which are named as strings, which means that
            these parameters are set as `variables`, the correspondences between these values and the true name of
            these parameters are stored as :obj:`dict` (:attr:`replace_params_dict`).
        """

        self.replace_params_dict = {}

        for key in params_dict.keys():
            if type(params_dict[key]) is str:
                if params_dict[key] in self._cond_var:
                    self.replace_params_dict[params_dict[key]] = key
                else:
                    raise ValueError("parameter setting {}:{} is not valid because cond_var does not contains {}."
                                     .format(key, params_dict[key], params_dict[key]))
            elif isinstance(params_dict[key], torch.Tensor):
                features = params_dict[key]
                features_checked = self._check_features_shape(features)
                # clone features to make it contiguous & to make it independent.
                self.register_buffer(key, features_checked.clone())
            else:
                raise ValueError()

    def _check_features_shape(self, features):
        # scalar
        if features.size() == torch.Size():
            features = features.expand(self.features_shape)

        if self.features_shape == torch.Size():
            self._features_shape = features.shape

        if features.size() == self.features_shape:
            batches = features.unsqueeze(0)
            return batches

        raise ValueError("the shape of a given parameter {} and features_shape {} "
                         "do not match.".format(features.size(), self.features_shape))

    @property
    def params_keys(self):
        """list: Return the list of parameter names for this distribution."""
        raise NotImplementedError()

    @property
    def distribution_torch_class(self):
        """Return the class of PyTorch distribution."""
        raise NotImplementedError()

    @property
    def dist(self):
        """Return the instance of PyTorch distribution."""
        return self._dist

    def set_dist(self, x_dict={}, batch_n=None, **kwargs):
        """Set :attr:`dist` as PyTorch distributions given parameters.

        This requires that :attr:`params_keys` and :attr:`distribution_torch_class` are set.

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}.
            Parameters of this distribution.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        params = self.get_params(x_dict, **kwargs)
        if set(self.params_keys) != set(params.keys()):
            raise ValueError("{} class requires following parameters: {}\n"
                             "but got {}".format(type(self), set(self.params_keys), set(params.keys())))

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

    def get_sample(self, reparam=False, sample_shape=torch.Size()):
        """Get a sample_shape shaped sample from :attr:`dist`.

        Parameters
        ----------
        reparam : :obj:`bool`, defaults to True.
            Choose where to sample using re-parameterization trick.

        sample_shape : :obj:`tuple` or :obj:`torch.Size`, defaults to torch.Size().
            Set the shape of a generated sample.

        Returns
        -------
        samples_dict : dict
            Generated sample formatted by :obj:`dict`.

        """
        if reparam and self.dist.has_rsample:
            _samples = self.dist.rsample(sample_shape=sample_shape)
        else:
            _samples = self.dist.sample(sample_shape=sample_shape)
        samples_dict = {self._var[0]: _samples}

        return samples_dict

    @property
    def has_reparam(self):
        raise NotImplementedError()

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        _x_dict = get_dict_values(x_dict, self._cond_var, return_dict=True)
        self.set_dist(_x_dict)

        x_targets = get_dict_values(x_dict, self._var)
        log_prob = self.dist.log_prob(*x_targets)
        if sum_features:
            log_prob = sum_samples(log_prob)

        return log_prob

    @lru_cache_for_sample_dict()
    def get_params(self, params_dict={}, **kwargs):
        """This method aims to get parameters of this distributions from constant parameters set in initialization
        and outputs of DNNs.

        Parameters
        ----------
        params_dict : :obj:`dict`, defaults to {}
            Input parameters.

        Returns
        -------
        output_dict : dict
            Output parameters.

        Examples
        --------
        >>> from pixyz.distributions import Normal
        >>> dist_1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...                 features_shape=[1])
        >>> print(dist_1)
        Distribution:
          p(x)
        Network architecture:
          p(x) =
          Normal(
            name=p, distribution_name=Normal,
            features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
        >>> dist_1.get_params()
        {'loc': tensor([[0.]]), 'scale': tensor([[1.]])}

        >>> dist_2 = Normal(loc=torch.tensor(0.), scale="z", cond_var=["z"], var=["x"])
        >>> print(dist_2)
        Distribution:
          p(x|z)
        Network architecture:
          p(x|z) =
          Normal(
            name=p, distribution_name=Normal,
            features_shape=torch.Size([])
            (loc): torch.Size([1])
          )
        >>> dist_2.get_params({"z": torch.tensor(1.)})
        {'scale': tensor(1.), 'loc': tensor([0.])}

        """
        params_dict, vars_dict = replace_dict_keys_split(params_dict, self.replace_params_dict)
        output_dict = self.forward(**vars_dict)

        output_dict.update(params_dict)

        # append constant parameters to output_dict
        constant_params_dict = get_dict_values(dict(self.named_buffers()), self.params_keys,
                                               return_dict=True)
        output_dict.update(constant_params_dict)

        return output_dict

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        _x_dict = get_dict_values(x_dict, self._cond_var, return_dict=True)
        self.set_dist(_x_dict)

        entropy = self.dist.entropy()
        if sum_features:
            entropy = sum_samples(entropy)

        return entropy

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        # check whether the input is valid or convert it to valid dictionary.
        input_dict = self._get_input_dict(x_dict)

        self.set_dist(input_dict, batch_n=batch_n)
        output_dict = self.get_sample(reparam=reparam, sample_shape=sample_shape)

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def sample_mean(self, x_dict={}):
        self.set_dist(x_dict)
        return self.dist.mean

    def sample_variance(self, x_dict={}):
        self.set_dist(x_dict)
        return self.dist.variance

    def forward(self, **params):
        return params

    @property
    def prob_factorized_text(self):
        """str: Return a formula of the factorized probability distribution."""
        return self.graph.prob_text


class MultiplyDistribution(Distribution):
    """Multiply by given distributions, e.g, :math:`p(x,y|z) = p(x|z,y)p(y|z)`.
    In this class, it is checked if two distributions can be multiplied.

    p(x|z)p(z|y) -> Valid

    p(x|z)p(y|z) -> Valid

    p(x|z)p(y|a) -> Valid

    p(x|z)p(z|x) -> Invalid (recursive)

    p(x|z)p(x|y) -> Invalid (conflict)

    Examples
    --------
    >>> a = DistributionBase(var=["x"], cond_var=["z"])
    >>> b = DistributionBase(var=["z"], cond_var=["y"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,z|y) = p(x|z)p(z|y)
    Network architecture:
      p(z|y) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"], cond_var=["z"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      p(y|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"], cond_var=["a"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z,a) = p(x|z)p(y|a)
    Network architecture:
      p(y|a) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )

    """

    def __init__(self, a, b):
        """
        Parameters
        ----------
        a : pixyz.Distribution
            Distribution.

        b : pixyz.Distribution
            Distribution.

        """
        super().__init__(var=[], atomic=False)
        self.graph = a.graph.united(b.graph)

    def __repr__(self):
        return repr(self.graph)


class ReplaceVarDistribution(Distribution):
    """Replace names of variables in Distribution.

    Examples
    --------
    >>> p = DistributionBase(var=["x"], cond_var=["z"])
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
    >>> replace_dict = {'x': 'y'}
    >>> p_repl = ReplaceVarDistribution(p, replace_dict)
    >>> print(p_repl)
    Distribution:
      p(y|z)
    Network architecture:
      p(y|z) -> p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )

    """

    def __init__(self, p, replace_dict):
        """
        Parameters
        ----------
        p : :class:`pixyz.distributions.Distribution` (not :class:`pixyz.distributions.MultiplyDistribution`)
            Distribution.

        replace_dict : dict
            Dictionary.

        """
        super().__init__(cond_var=[], var=[], name=p.name, features_shape=p.features_shape, atomic=False)
        self.graph = p.graph.var_replaced(replace_dict)
        self.p = p

    def __repr__(self):
        return repr(self.graph)

    def forward(self, *args, **kwargs):
        return self.p.forward(*args, **kwargs)

    def sample_mean(self, x_dict={}):
        return self.p.sample_mean(x_dict)

    def sample_variance(self, x_dict={}):
        return self.p.sample_variance(x_dict)

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        return self.p.get_entropy(x_dict, sum_features, feature_dims)

    @property
    def distribution_name(self):
        return self.p.distribution_name

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self.p.__getattribute__(item)


class MarginalizeVarDistribution(Distribution):
    r"""Marginalize variables in Distribution.

    .. math::
        p(x) = \int p(x,z) dz

    Examples
    --------
    >>> a = DistributionBase(var=["x"], cond_var=["z"])
    >>> b = DistributionBase(var=["y"], cond_var=["z"])
    >>> p_multi = a * b
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      p(y|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
    >>> p_marg = MarginalizeVarDistribution(p_multi, ["y"])
    >>> print(p_marg)
    Distribution:
      p(x|z) = \int p(x|z)p(y|z)dy
    Network architecture:
      p(y|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )
      p(x|z) =
      DistributionBase(
        name=p, distribution_name=,
        features_shape=torch.Size([])
      )

    """

    def __init__(self, p: Distribution, marginalize_list):
        """
        Parameters
        ----------
        p : :class:`pixyz.distributions.Distribution` (not :class:`pixyz.distributions.DistributionBase`)
            Distribution.

        marginalize_list : list
            Variables to marginalize.

        """
        marginalize_list = tolist(marginalize_list)

        super().__init__(cond_var=[], var=[], name=p.name, features_shape=p.features_shape, atomic=False)
        self.graph = p.graph.marginalized(marginalize_list)
        self.p = p

    def __repr__(self):
        return repr(self.graph)

    def forward(self, *args, **kwargs):
        return self.p.forward(*args, **kwargs)

    def sample_mean(self, x_dict={}):
        return self.p.sample_mean(x_dict)

    def sample_variance(self, x_dict={}):
        return self.p.sample_variance(x_dict)

    def get_entropy(self, x_dict={}, sum_features=True, feature_dims=None):
        return self.p.get_entropy(x_dict, sum_features, feature_dims)

    @property
    def distribution_name(self):
        return self.p.distribution_name

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self.p.__getattribute__(item)
