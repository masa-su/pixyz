from __future__ import print_function
import torch
import re
import networkx as nx
from torch import nn

from ..utils import get_dict_values, replace_dict_keys, delete_dict_values,\
    tolist, sum_samples, convert_latex_name, lru_cache_for_sample_dict
from ..losses import LogProb, Prob


def _make_prob_text(dist_name, var, cond_var):
    var_text = ','.join(convert_latex_name(var_name) for var_name in var)
    cond_text = '' if len(cond_var) == 0 else \
        '|' + ','.join(convert_latex_name(var_name) for var_name in cond_var)
    return f"{dist_name}({var_text}{cond_text})"


def _make_prob_equality_text(prob_text, prob_factorized_text):
    if prob_factorized_text == prob_text:
        return prob_text
    else:
        return f"{prob_text} = {prob_factorized_text}"


def _make_distribution_text(prob_joint_factorized_and_text, network_text):
    # Distribution
    text = f"Distribution:\n  {prob_joint_factorized_and_text}\n"

    # Network architecture (`repr`)
    network_text = re.sub('^', ' ' * 2, str(network_text), flags=re.MULTILINE)
    text += f"Network architecture:\n{network_text}"
    return text


class Factor:
    """
    This class wraps an atomic distribution as a factor node of a DistGraph.
    It allocates new instance even if the same atomic distribution is specified.
    This class assumes the lifespan of it is covered by the lifespan of the DistGraph.
    """
    def __init__(self, atom_dist):
        self.dist = atom_dist
        self.name_dict = {}
        self.option = {}

    def copy(self):
        inst = Factor(self.dist)
        inst.name_dict = dict(self.name_dict)
        inst.option = dict(self.option)
        return inst

    def rename_var(self, replace_dict):
        name_dict = self.name_dict
        # name_dict:global->local + replace:global->new_global = name_dict:new_global->local
        for var_name, new_var_name in replace_dict.items():
            if var_name in name_dict:
                local_var = name_dict[var_name]
                del name_dict[var_name]
                name_dict[new_var_name] = local_var
            else:
                name_dict[new_var_name] = var_name

    @property
    def _reversed_name_dict(self):
        return {value: key for key, value in self.name_dict.items()}

    @staticmethod
    def __apply_dict(dict, var):
        return [dict[var_name] if var_name in dict else var_name for var_name in var]

    def sample(self, values, sample_option):
        global_input_var = self.__apply_dict(self._reversed_name_dict, self.dist.input_var)

        if any(var_name not in values for var_name in global_input_var):
            raise ValueError("lack of some condition variables")
        input_dict = get_dict_values(values, global_input_var, return_dict=True)

        local_input_dict = replace_dict_keys(input_dict, self.name_dict)

        # Overwrite log_prob_option with self.option to give priority to local settings such as batch_n
        option = dict(sample_option)
        option.update(self.option)
        local_output_dict = self.dist.sample(local_input_dict, **option)

        # TODO: It shows return_hidden option change graphical model. This is bad operation.
        ignore_hidden = ('return_hidden' in sample_option and sample_option['return_hidden'])
        ignore_hidden |= ('return_hidden' in self.option and self.option['return_hidden'])
        if not ignore_hidden and set(local_output_dict) != set(self.dist.var):
            raise Exception(f"The sample method of {self.dist.distribution_name} returns different variables."
                            f" Expected:{list(self.dist.var)}, Got:{list(local_output_dict)}")

        sample = replace_dict_keys(local_output_dict, self._reversed_name_dict)
        return sample

    def get_log_prob(self, values, log_prob_option):
        global_input_var = self.__apply_dict(self._reversed_name_dict, list(self.dist.var) + list(self.dist.cond_var))

        if any(var_name not in values for var_name in global_input_var):
            raise ValueError("lack of some variables")
        input_dict = get_dict_values(values, global_input_var, return_dict=True)
        local_input_dict = replace_dict_keys(input_dict, self.name_dict)

        # Overwrite log_prob_option with self.option to give priority to local settings such as batch_n
        option = dict(log_prob_option)
        option.update(self.option)
        log_prob = self.dist.get_log_prob(local_input_dict, **option)
        return log_prob

    @property
    def input_var(self):
        return self.__apply_dict(self._reversed_name_dict, self.dist.input_var)

    @property
    def var(self):
        return self.__apply_dict(self._reversed_name_dict, self.dist.var)

    @property
    def cond_var(self):
        return self.__apply_dict(self._reversed_name_dict, self.dist.cond_var)

    @property
    def prob_text(self):
        return _make_prob_text(self.dist.name, self.var, self.cond_var)

    def __str__(self):
        prob_node_text = self.prob_text
        factorized_text = self.dist.prob_factorized_text
        if prob_node_text == factorized_text:
            header_text = f"{prob_node_text}:\n"
        else:
            header_text = f"{prob_node_text} -> {self.dist.prob_joint_factorized_and_text}:\n"
        return header_text + repr(self.dist)


class DistGraph(nn.Module):
    """
     Graphical model class. This manages the graph of Graphical Model of distribution.
     It is called from Distribution class.
    """
    def __init__(self, original=None):
        super().__init__()
        self.graph = nx.DiGraph()
        self.global_option = {}
        self.marginalize_list = set()
        self.name = ''
        if original:
            self._override_module(original)
            self.graph = nx.relabel_nodes(original.graph,
                                          {factor: factor.copy() for factor in original.factors()})
            self.global_option.update(original.global_option)
            self.marginalize_list.update(original.marginalize_list)
            self.name = original.name

    def _override_module(self, original: nn.Module):
        name_offset = len(list(self.named_children()))
        for i, (_, module) in enumerate(original.named_children()):
            self.add_module(str(name_offset + i), module)

    def appended(self, atom_dist):
        """ Return new graph appended one node.
        Parameters
        ----------
        atom_dist : Distribution

        Returns
        -------
        DistGraph

        """
        new_instance = DistGraph(self)
        if not new_instance.name:
            new_instance.name = atom_dist.name
        # factor node of an atomic distribution
        factor = Factor(atom_dist)
        new_instance.add_module(str(len(list(new_instance.factors()))), atom_dist)
        new_instance.graph.add_node(factor)
        for var_name in atom_dist.var:
            if var_name in new_instance.graph:
                raise ValueError(f"A new variable name '{var_name}' is already used in this graph.")
            new_instance.graph.add_edge(factor, var_name)
        for cond in atom_dist.cond_var:
            new_instance.graph.add_edge(cond, factor)
        return new_instance

    def set_option(self, option_dict, var=[]):
        """ Set option arguments which used when you call `sample` or `get_log_prob` methods.
        Parameters
        ----------
        option_dict: dict of str and any object
        var: list of string
        Examples
        --------
        >>> from pixyz.distributions import Normal
        >>> dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1)
        >>> # Set options only on the sampling start node
        >>> dist.graph.set_option(dict(batch_n=4, sample_shape=(2, 3)), ['y'])
        >>> sample = dist.sample()
        >>> sample['y'].shape
        torch.Size([2, 3, 4])
        >>> sample['x'].shape
        torch.Size([2, 3, 4])
        """
        if not var:
            self.global_option = option_dict
        else:
            for var_name in var:
                for factor in self._factors_from_variable(var_name):
                    factor.option = option_dict

    def united(self, other):
        if not set(self.var + list(self.marginalize_list)).isdisjoint(set(other.var + list(other.marginalize_list))):
            raise ValueError("There is var-name conflicts between two graphs.")
        if not set(self.factors()).isdisjoint(set(other.factors())):
            raise ValueError("The same instances of a distribution are used between two graphs.")
        scg = DistGraph(self)
        scg._override_module(other)
        scg.graph.update(other.graph)
        scg.global_option.update(other.global_option)
        scg.marginalize_list.update(other.marginalize_list)
        return scg

    def marginalized(self, marginalize_list):
        """ Return new graph marginalized some variables
        Parameters
        ----------
        marginalize_list : iterative of str

        Returns
        -------
        DistGraph

        Examples
        --------
        >>> import pixyz.distributions as pd
        >>> dist = pd.Normal(var=['x']).marginalize_var(['x'])
        Traceback (most recent call last):
        ...
        ValueError: marginalize_list has unknown variables or it has all of variables of `p`.
        >>> dist = (pd.Normal(var=['x'])*pd.Normal(var=['y'])).marginalize_var(['x'])
        >>> dist.graph.marginalize_list
        {'x'}
        >>> dist.var
        ['y']
        >>> dist.cond_var
        []

        """
        marginalize_list = set(marginalize_list)
        if len(marginalize_list) == 0:
            raise ValueError("Length of `marginalize_list` must be at least 1, got 0.")
        if not marginalize_list < set(self.var):
            raise ValueError("marginalize_list has unknown variables or it has all of variables of `p`.")

        new_graph = DistGraph(self)
        new_graph.marginalize_list.update(marginalize_list)
        return new_graph

    def var_replaced(self, replace_dict):
        r""" Returns new graph whose variables are replaced.
        Parameters
        ----------
        replace_dict: dict of str and str

        Returns
        -------
        DistGraph

        Examples
        --------
        >>> from pixyz.distributions.distributions import DistGraph
        >>> import pixyz.distributions as pd
        >>> normal = pd.Normal(var=['x'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> normal2 = pd.Normal(var=['y'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> multi_dist = normal * normal2
        >>> normal3 = pd.Normal(var=['z'], cond_var=['y'], loc='y', scale=torch.ones(1))
        >>> multi_dist2 = multi_dist * normal3
        >>> # 周辺化した変数へのリネームは許可しない
        >>> dist3 = multi_dist2.marginalize_var(['y']).replace_var(z='y')
        Traceback (most recent call last):
        ...
        ValueError: ['y', 'z'] are conflicted after replaced.
        >>> dist3 = multi_dist2.marginalize_var(['y']).replace_var(z='w', x='z')
        >>> sample = dist3.sample()
        >>> sample # doctest: +SKIP
        {'w': tensor([[2.3206]]), 'z': tensor([[-0.5381]])}
        >>> dist4 = multi_dist2.marginalize_var(['y']).replace_var(z='w', x='z').replace_var(z='a')
        >>> print(dist4)
        Distribution:
          p(w,a) = \int p(a)p(w|y)p(y)dy
        Network architecture:
          p(y):
          Normal(
            name=p, distribution_name=Normal,
            var=['y'], cond_var=[], input_var=[], features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
          p(w|y) -> p(z|y):
          Normal(
            name=p, distribution_name=Normal,
            var=['z'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([1])
            (scale): torch.Size([1, 1])
          )
          p(a) -> p(x):
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
        >>> print(repr(dist4))
        DistGraph(
          (0): Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
          (1): Normal(
            name=p, distribution_name=Normal,
            var=['y'], cond_var=[], input_var=[], features_shape=torch.Size([1])
            (loc): torch.Size([1, 1])
            (scale): torch.Size([1, 1])
          )
          (2): Normal(
            name=p, distribution_name=Normal,
            var=['z'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([1])
            (scale): torch.Size([1, 1])
          )
        )
        """
        # check replace_dict
        if not (set(replace_dict) <= set(self.all_var)):
            unknown_var = [var_name for var_name in replace_dict.keys() if var_name not in self.all_var]
            raise ValueError(f"replace_dict has unknown variables: {unknown_var}")
        replaced_vars = [replace_dict[var_name] if var_name in replace_dict else var_name for var_name in self.all_var]
        if len(self.all_var) != len(set(replaced_vars)):
            duplicated_vars = [var_name for var_name in self.all_var
                               if replaced_vars.count(replace_dict[var_name]
                                                      if var_name in replace_dict else var_name) > 1]
            raise ValueError(f"{duplicated_vars} are conflicted after replaced.")

        result = DistGraph(original=self)
        result.graph = nx.relabel_nodes(result.graph, replace_dict, copy=False)
        result.marginalize_list = {replace_dict[var] if var in replace_dict else var for var in self.marginalize_list}
        result.global_option = dict(self.global_option)

        for factor in result.factors():
            if set(replace_dict.values()).isdisjoint(list(result.graph.pred[factor]) + list(result.graph.succ[factor])):
                continue
            factor.rename_var(replace_dict)
        return result

    def _factors_from_variable(self, var_name):
        return list(self.graph.pred[var_name])

    def factors(self, sorted=False):
        """ get factors of the DistGraph.
        Parameters
        ----------
        sorted: bool
            the order of factors is topological sorted or not.

        Returns
        -------
        iter of Factor

        """
        nodes = nx.topological_sort(self.graph) if sorted else self.graph
        for node in nodes:
            if isinstance(node, Factor):
                yield node

    def distribution(self, var_name):
        """ An atomic distribution of the specified variable.
        Parameters
        ----------
        var_name: str

        Returns
        -------
        Distribution
        """
        factors = self._factors_from_variable(var_name)
        if len(factors) == 0:
            raise ValueError(f"There is no distirbution about {var_name}.")
        if len(factors) != 1:
            raise NotImplementedError("multiple factors are not supported now.")
        return factors[0].dist

    @property
    def all_var(self):
        """ All variables in the DistGraph.
        Returns
        -------
        list of str
        """
        return [var_name for var_name in self.graph if isinstance(var_name, str)]

    @property
    def input_var(self):
        """ conditional variables and observation variables in the DistGraph.
        Returns
        -------
        list of str
        """
        def is_input_var_node(var_name):
            if not isinstance(var_name, str):
                return False
            if not self.graph.pred[var_name]:
                return True
            if var_name in self._factors_from_variable(var_name)[0].input_var:
                return True
            else:
                return False
        return [var_name for var_name in self.graph if is_input_var_node(var_name)]

    @property
    def cond_var(self):
        """ conditional variables in the DistGraph.
        Returns
        -------
        list of str
        """
        return [var_name for var_name in self.graph if isinstance(var_name, str) and not self.graph.pred[var_name]]

    @property
    def var(self):
        """ hidden variables in the DistGraph.
        Returns
        -------
        list of str
        """
        def is_var_node(var_name):
            if not isinstance(var_name, str):
                return False
            if self.graph.pred[var_name] and var_name not in self.marginalize_list:
                return True
            else:
                return False
        return [var_name for var_name in self.graph if is_var_node(var_name)]

    def forward(self, mode, kwargs):
        if mode == 'sample':
            return self._sample(**kwargs)
        elif mode == 'get_log_prob':
            return self._get_log_prob(**kwargs)
        else:
            raise ValueError()

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
               sample_mean=False, **kwargs):
        _kwargs = dict(x_dict=x_dict, batch_n=batch_n, sample_shape=sample_shape,
                       return_all=return_all, reparam=reparam, sample_mean=sample_mean)
        _kwargs.update(kwargs)
        return self('sample', kwargs=_kwargs)

    def _sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
                sample_mean=False, **kwargs):
        """
        Sample variables of this distribution.
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
        >>> from pixyz.distributions.distributions import DistGraph
        >>> import pixyz.distributions as pd
        >>> # atomへのアクセスにはgraphは使われない．
        >>> normal = pd.Normal(var=['x'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> normal.sample(batch_n=2, sample_shape=torch.Size((3, 4)),
        ...     return_all=True, reparam=True)['x'].shape
        torch.Size([3, 4, 2, 1])
        >>> normal2 = pd.Normal(var=['y'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> multi_dist = normal * normal2
        >>> sample = multi_dist.sample()
        >>> sample # doctest: +SKIP
        {'y': tensor([[0.6635]]), 'x': tensor([[0.3966]])}
        >>> sample = multi_dist.sample(batch_n=2)
        >>> normal3 = pd.Normal(var=['z'], cond_var=['y'], loc='y', scale=torch.ones(1))
        >>> wrong_dist = multi_dist * normal2
        Traceback (most recent call last):
        ...
        ValueError: There is var-name conflicts between two graphs.
        >>> multi_dist2 = multi_dist * normal3
        >>> # TODO: this issue will be solved at another pull request. distribution with cond_var has the problem.
        >>> multi_dist2.sample(batch_n=2, sample_shape=(3, 4))
        Traceback (most recent call last):
        ...
        ValueError: Batch shape mismatch. batch_shape from parameters: torch.Size([3, 4, 2, 1])
         specified batch size:2
        >>> sample = multi_dist2.sample(batch_n=2)
        >>> sample # doctest: +SKIP
        {'y': tensor([[1.6723], [0.1929]]), 'z': tensor([[ 0.8572], [-0.5933]]), 'x': tensor([[-0.4255], [-0.4793]])}
        >>> sample = multi_dist2.sample(sample_shape=(1,))
        >>> sample # doctest: +SKIP
        {'y': tensor([[[-0.8537]]]), 'z': tensor([[[[-2.1819]]]]), 'x': tensor([[[-0.0797]]])}
        >>> # return_all=Falseで条件付けられた変数や使用しなかった変数を含まない戻り値を得る
        >>> normal4 = pd.Normal(var=['a'], cond_var=['b'], loc='b', scale=torch.ones(1))
        >>> dist3 = multi_dist2.marginalize_var(['y']).replace_var(z='w').replace_var(x='z').replace_var(z='x')*normal4
        >>> sample = dist3.sample(x_dict={'b': torch.ones(2, 1), 'c': torch.zeros(1)}, return_all=False)
        >>> sample.keys()
        dict_keys(['a', 'w', 'x'])
        >>> from pixyz.distributions import Normal, Categorical
        >>> from pixyz.distributions.mixture_distributions import MixtureModel
        >>> z_dim = 3  # the number of mixture
        >>> x_dim = 2  # the input dimension.
        >>> distributions = []  # the list of distributions
        >>> for i in range(z_dim):
        ...     loc = torch.randn(x_dim)  # initialize the value of location (mean)
        ...     scale = torch.empty(x_dim).fill_(1.)  # initialize the value of scale (variance)
        ...     distributions.append(Normal(loc=loc, scale=scale, var=["y"], name="p_%d" %i))
        >>> probs = torch.empty(z_dim).fill_(1. / z_dim)  # initialize the value of probabilities
        >>> prior = Categorical(probs=probs, var=["z"], name="prior")
        >>> p = MixtureModel(distributions=distributions, prior=prior)
        >>> dist = normal*p
        >>> dist.graph.set_option({'return_hidden': True}, var=['y'])
        >>> list(dist.sample().keys())
        ['y', 'z', 'x']

        """

        sample_option = dict(self.global_option)
        sample_option.update(dict(batch_n=batch_n, sample_shape=sample_shape,
                                  return_all=False, reparam=reparam, sample_mean=sample_mean))
        sample_option.update(kwargs)
        # ignore return_all because overriding is now under control.
        if not(set(x_dict) >= set(self.input_var)):
            raise ValueError(f"Input keys are not valid, expected {set(self.input_var)} but got {set(x_dict)}.")

        values = get_dict_values(x_dict, self.input_var, return_dict=True)
        for factor in self.factors(sorted=True):
            sample = factor.sample(values, sample_option)
            values.update(sample)

        result_dict = delete_dict_values(values, self.marginalize_list)
        if return_all:
            output_dict = dict(delete_dict_values(x_dict, self.input_var))
            output_dict.update(result_dict)
            return output_dict
        else:
            return delete_dict_values(result_dict, self.input_var)

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        return self(mode='get_log_prob', kwargs={'x_dict': x_dict, 'sum_features': sum_features,
                                                 'feature_dims': feature_dims})

    def _get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        """ Giving variables, this method returns values of log-pdf.

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
        >>> from pixyz.distributions.distributions import DistGraph
        >>> import torch
        >>> import pixyz.distributions as pd
        >>> # atomへのアクセスにはgraphは使われない．
        >>> pd.Normal(var=['x'], loc=torch.zeros(1), scale=torch.ones(1)).get_log_prob({'x': torch.zeros(1, 1)})
        tensor([-0.9189])
        >>> # 同時分布などにはDistGraphが使われる
        >>> dist = pd.Normal(var=['x'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> dist *= pd.Normal(var=['y'], loc=torch.zeros(1), scale=torch.ones(1))
        >>> dist = dist.replace_var(y='z')
        >>> dist.get_log_prob({'x': torch.zeros(1, 1), 'z': torch.zeros(1, 1)})
        tensor([-1.8379])
        >>> # 周辺化がある場合，対数尤度は計算されない．
        >>> m_dist = dist.marginalize_var(['z'])
        >>> m_dist.get_log_prob({'x': torch.zeros(1, 1)})
        Traceback (most recent call last):
        ...
        NotImplementedError
        """
        # """
        # >>> # 確率変数の周辺化がある場合，対数尤度は計算されない．
        # >>> m_dist = dist.marginalize_var(['z'])
        # >>> m_dist.get_log_prob({'x': torch.zeros(1, 1)})
        # Traceback (most recent call last):
        # ...
        # ValueError: This distribution is marginalized by the stochastic variables '['z']'. Log probability of it can not be calcurated.
        # >>> # 決定論的な変数の周辺化がある場合，決定論的な変数が一致する前提で対数尤度が計算される．
        # >>> class MyDeterministic(pd.Deterministic):
        # ...     def forward(self):
        # ...         return {'x': torch.zeros(1, 1)}
        # >>> dist = MyDeterministic(var=['x'])
        # >>> dist *= pd.Normal(var=['y'], cond_var=['x'], loc='x', scale=torch.ones(1))
        # >>> dist.get_log_prob({'y': torch.zeros(1, 1), 'x': torch.zeros(1, 1)})
        # Traceback (most recent call last):
        # ...
        # NotImplementedError: Log probability of deterministic distribution is not defined.
        # >>> m_dist = dist.marginalize_var(['x'])
        # >>> m_dist.get_log_prob({'y': torch.zeros(1, 1)})
        # tensor([-0.9189])
        # """

        sample_option = dict(self.global_option)
        # sample_option.update(dict(batch_n=batch_n, sample_shape=sample_shape, return_all=False))

        if len(self.marginalize_list) != 0:
            raise NotImplementedError()

        log_prob_option = dict(self.global_option)
        log_prob_option.update(dict(sum_features=sum_features, feature_dims=feature_dims))
        log_prob_option.update(kwargs)

        require_var = self.var + self.cond_var
        if not(set(x_dict) >= set(require_var)):
            raise ValueError(f"Input keys are not valid, expected {set(require_var)}"
                             f" but got {set(x_dict)}.")

        values = get_dict_values(x_dict, require_var, return_dict=True)
        log_prob = None
        prev_dist = None
        for factor in self.factors(sorted=True):
            local_var = self.graph.succ[factor]

            local_marginalized_var = [var_name for var_name in local_var if var_name in self.marginalize_list]
            if len(local_marginalized_var) != 0:
                if any(var_name in values for var_name in local_marginalized_var):
                    raise ValueError(f"The marginalized variables '{local_marginalized_var}'"
                                     f" appears in the dictionary: {x_dict}.")
                if factor.dist.distribution_name != "Deterministic":
                    raise ValueError(f"This distribution is marginalized by the stochastic variables '{local_marginalized_var}'."
                                     f" Log probability of it can not be calcurated.")
                if set(local_var) != set(local_marginalized_var):
                    raise ValueError("Some deterministic variables are not marginalized.")
                # batch_nに関しては後続の変数に与えられた値で判断できる，sample_shapeはnamed_shapeなら解決できそう
                sample = factor.sample(values, sample_option)
                values.update(sample)
                continue

            new_log_prob = factor.get_log_prob(values, log_prob_option)

            if log_prob is None:
                log_prob = new_log_prob
            else:
                if log_prob.size() != new_log_prob.size():
                    raise ValueError(f"Two PDFs, {prev_dist.prob_text} and {factor.dist.prob_text}, have different sizes,"
                                     " so you must modify these tensor sizes.")
                log_prob += new_log_prob
            prev_dist = factor.dist
        if log_prob is None:
            return 0
        return log_prob

    @property
    def has_reparam(self):
        return all(factor.dist.has_reparam for factor in self.factors())

    def __str__(self):
        network_text = "\n".join(str(factor) for factor in self.factors(sorted=True))
        return _make_distribution_text(self.prob_joint_factorized_and_text, network_text)

    @property
    def prob_text(self):
        return _make_prob_text(self.name, self.var, self.cond_var)

    @property
    def prob_factorized_text(self):
        text = ""
        for factor in self.factors(sorted=True):
            text = factor.prob_text + text
        if self.marginalize_list:
            integral_symbol = len(self.marginalize_list) * "\\int "
            integral_variables = ["d" + convert_latex_name(var) for var in self.marginalize_list]
            integral_variables = "".join(integral_variables)

            return f"{integral_symbol}{text}{integral_variables}"
        return text

    @property
    def prob_joint_factorized_and_text(self):
        return _make_prob_equality_text(self.prob_text, self.prob_factorized_text)

    def visible_graph(self, dotmode=False):
        visible_graph = nx.DiGraph()

        def dont_esc(name: str):
            return f"${name}$"
        for factor in self.factors():
            for var_name in factor.var:
                for cond_var_name in factor.cond_var:
                    if dotmode:
                        visible_graph.add_edge(cond_var_name, var_name)
                    else:
                        visible_graph.add_edge(dont_esc(cond_var_name), dont_esc(var_name))
        if dotmode:
            for var_name in visible_graph:
                visible_graph.add_node(var_name, texlbl=dont_esc(var_name))
        return visible_graph


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
      Normal(
        name=p_{1}, distribution_name=Normal,
        var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([64])
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
      Normal(
        name=p_{2}, distribution_name=Normal,
        var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([64])
        (scale): torch.Size([1, 64])
      )

    >>> # Conditional distribution (by neural networks)
    >>> class P(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["x"],cond_var=["y"],name="p3")
    ...         self.model_loc = nn.Linear(128, 64)
    ...         self.model_scale = nn.Linear(128, 64)
    ...     def forward(self, y):
    ...         return {"loc": self.model_loc(y), "scale": F.softplus(self.model_scale(y))}
    >>> p3 = P()
    >>> print(p3)
    Distribution:
      p_{3}(x|y)
    Network architecture:
      P(
        name=p_{3}, distribution_name=Normal,
        var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([])
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
        self._name = convert_latex_name(name)

        self._atomic = atomic
        if atomic and len(var) == 0:
            raise ValueError("At least one variable is required for an atomic distribution.")
        self._graph = None

        self._features_shape = torch.Size(features_shape)

    @property
    def graph(self):
        if self._atomic:
            if not self._graph:
                # (graph,) for escaping meta-language of nn.Module
                self._graph = (DistGraph().appended(atom_dist=self),)
            return self._graph[0]
        else:
            return self._graph

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
            if self._atomic:
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

        return _make_prob_text(self._name, self.var, self.cond_var)

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

        return _make_prob_equality_text(self.prob_text, self.prob_factorized_text)

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
            if not (set(input) >= set(var)):
                raise ValueError(f"Input keys are not valid, expected {set(var)} but got {set(input)}.")
            input_dict = get_dict_values(input, var, return_dict=True)

        else:
            raise ValueError("The type of input is not valid, got %s." % type(input))

        return input_dict

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True,
               reparam=False, sample_mean=False, **kwargs):
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
        >>> p = Normal(loc=0, scale=1, var=["x"], features_shape=[10, 2])
        >>> print(p)
        Distribution:
          p(x)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([10, 2])
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
        >>> p = Normal(loc="y", scale=1., var=["x"], cond_var=["y"], features_shape=[10])
        >>> print(p)
        Distribution:
          p(x|y)
        Network architecture:
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([10])
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
            return self.graph.sample(x_dict, batch_n, sample_shape, return_all, reparam, sample_mean, **kwargs)
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

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
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
            return self.graph.get_log_prob(x_dict, sum_features, feature_dims, **kwargs)
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
        """Return an instance of :class:`pixyz.losses.Prob`.

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
        if not self._atomic:
            return str(self.graph)

        network_text = self.__repr__()
        return _make_distribution_text(self.prob_joint_factorized_and_text, network_text)

    def extra_repr(self):
        # parameters
        parameters_text = f'name={self.name}, distribution_name={self.distribution_name},\n' \
                          f'var={self.var}, cond_var={self.cond_var}, input_var={self.input_var}, ' \
                          f'features_shape={self.features_shape}'

        if len(self._buffers) != 0:
            # add buffers to repr
            buffers = [f"({key}): {value.shape}" for key, value in self._buffers.items()]
            return parameters_text + "\n" + "\n".join(buffers)

        return parameters_text


class DistributionBase(Distribution):
    """Distribution class with PyTorch. In Pixyz, all distributions are required to inherit this class."""

    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), **kwargs):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape)

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
        for key, value in params_dict.items():
            if type(value) is str:
                if value in self._cond_var:
                    if value not in self.replace_params_dict:
                        self.replace_params_dict[value] = []
                    self.replace_params_dict[value].append(key)
                else:
                    raise ValueError(f"parameter setting {key}:{value} is not valid"
                                     f" because cond_var does not contains {value}.")
            elif isinstance(value, torch.Tensor) \
                    or isinstance(value, float) or isinstance(value, int):
                if not isinstance(value, torch.Tensor):
                    features = torch.tensor(value, dtype=torch.float)
                else:
                    features = value
                features_checked = self._check_features_shape(features)
                # clone features to make it contiguous & to make it independent.
                self.register_buffer(key, features_checked.clone())
            else:
                raise ValueError(f"The types that can be specified as parameters of distribution"
                                 f" are limited to str & torch.Tensor. Got: {type(value)}")

    def _check_features_shape(self, features):
        # scalar
        if features.size() == torch.Size():
            features = features.expand(self.features_shape)

        if self.features_shape == torch.Size():
            self._features_shape = features.shape

        if features.size() == self.features_shape:
            batches = features.unsqueeze(0)
            return batches

        raise ValueError(f"the shape of a given parameter {features.size()}"
                         f" and features_shape {self.features_shape} do not match.")

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
            raise ValueError(f"{type(self)} class requires following parameters: {set(self.params_keys)}\n"
                             f"but got {set(params.keys())}")

        self._dist = self.distribution_torch_class(**params)

        # expand batch_n
        if batch_n:
            batch_shape = self._dist.batch_shape
            if batch_shape[0] == 1:
                self._dist = self._dist.expand(torch.Size([batch_n]) + batch_shape[1:])
            elif batch_shape[0] == batch_n:
                return
            else:
                raise ValueError(f"Batch shape mismatch. batch_shape from parameters: {batch_shape}\n"
                                 f" specified batch size:{batch_n}")

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

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        _x_dict = get_dict_values(x_dict, self._cond_var, return_dict=True)
        self.set_dist(_x_dict)

        x_targets = get_dict_values(x_dict, self._var)
        if len(x_targets) == 0:
            raise ValueError(f"x_dict has no value of the stochastic variable. x_dict: {x_dict}")
        log_prob = self.dist.log_prob(*x_targets)
        if sum_features:
            log_prob = sum_samples(log_prob, feature_dims)

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
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=[], input_var=[], features_shape=torch.Size([1])
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
          Normal(
            name=p, distribution_name=Normal,
            var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
            (loc): torch.Size([1])
          )
        >>> dist_2.get_params({"z": torch.tensor(1.)})
        {'scale': tensor(1.), 'loc': tensor([0.])}

        """
        replaced_params_dict = {}
        for key, value in params_dict.items():
            if key in self.replace_params_dict:
                for replaced_key in self.replace_params_dict[key]:
                    replaced_params_dict[replaced_key] = value

        vars_dict = {key: value for key, value in params_dict.items() if key not in self.replace_params_dict}
        output_dict = self(**vars_dict)

        output_dict.update(replaced_params_dict)

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
            entropy = sum_samples(entropy, feature_dims)

        return entropy

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
               sample_mean=False, **kwargs):
        # check whether the input is valid or convert it to valid dictionary.
        input_dict = self._get_input_dict(x_dict)

        self.set_dist(input_dict, batch_n=batch_n)

        if sample_mean:
            mean = self.dist.mean
            if sample_shape != torch.Size():
                unsqueeze_shape = torch.Size([1] * len(sample_shape))
                unrepeat_shape = torch.Size([1] * mean.ndim)
                mean = mean.reshape(unsqueeze_shape + mean.shape).repeat(sample_shape + unrepeat_shape)
            output_dict = {self._var[0]: mean}
        else:
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
    >>> a = DistributionBase(var=["x"],cond_var=["z"])
    >>> b = DistributionBase(var=["z"],cond_var=["y"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,z|y) = p(x|z)p(z|y)
    Network architecture:
      p(z|y):
      DistributionBase(
        name=p, distribution_name=,
        var=['z'], cond_var=['y'], input_var=['y'], features_shape=torch.Size([])
      )
      p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"],cond_var=["z"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      p(y|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> b = DistributionBase(var=["y"],cond_var=["a"])
    >>> p_multi = MultiplyDistribution(a, b)
    >>> print(p_multi)
    Distribution:
      p(x,y|z,a) = p(x|z)p(y|a)
    Network architecture:
      p(y|a):
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['a'], input_var=['a'], features_shape=torch.Size([])
      )
      p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
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
        self._graph = a.graph.united(b.graph)

    def __repr__(self):
        return repr(self.graph)


class ReplaceVarDistribution(Distribution):
    """Replace names of variables in Distribution.

    Examples
    --------
    >>> p = DistributionBase(var=["x"],cond_var=["z"])
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> replace_dict = {'x': 'y'}
    >>> p_repl = ReplaceVarDistribution(p, replace_dict)
    >>> print(p_repl)
    Distribution:
      p(y|z)
    Network architecture:
      p(y|z) -> p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
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
        super().__init__(var=[], cond_var=[], name=p.name, features_shape=p.features_shape, atomic=False)
        self._graph = p.graph.var_replaced(replace_dict)
        self.p = p

    def __repr__(self):
        return repr(self.graph)

    def forward(self, *args, **kwargs):
        return self.p(*args, **kwargs)

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
            import warnings
            warnings.warn("this magic method will be deprecated.")
            return self.p.__getattribute__(item)


class MarginalizeVarDistribution(Distribution):
    r"""Marginalize variables in Distribution.

    .. math::
        p(x) = \int p(x,z) dz

    Examples
    --------
    >>> a = DistributionBase(var=["x"],cond_var=["z"])
    >>> b = DistributionBase(var=["y"],cond_var=["z"])
    >>> p_multi = a * b
    >>> print(p_multi)
    Distribution:
      p(x,y|z) = p(x|z)p(y|z)
    Network architecture:
      p(y|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
    >>> p_marg = MarginalizeVarDistribution(p_multi, ["y"])
    >>> print(p_marg)
    Distribution:
      p(x|z) = \int p(x|z)p(y|z)dy
    Network architecture:
      p(y|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['y'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
      )
      p(x|z):
      DistributionBase(
        name=p, distribution_name=,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
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

        super().__init__(var=[], cond_var=[], name=p.name, features_shape=p.features_shape, atomic=False)
        self._graph = p.graph.marginalized(marginalize_list)
        self.p = p

    def __repr__(self):
        return repr(self.graph)

    def forward(self, *args, **kwargs):
        return self.p(*args, **kwargs)

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
            import warnings
            warnings.warn("this magic method will be deprecated.")
            return self.p.__getattribute__(item)
