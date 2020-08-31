import functools
import itertools
import numpy as np
import torch
import networkx as nx
from matplotlib import rc
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import sympy
from IPython.display import Math
import pixyz

_EPSILON = 1e-07
CACHE_SIZE = 0

rc("font", family="serif", size=12)
rc("text", usetex=True)


def set_epsilon(eps):
    """Set a `epsilon` parameter.

    Parameters
    ----------
    eps : int or float

    Returns
    -------

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._EPSILON', 1e-07):
    ...     set_epsilon(1e-06)
    ...     epsilon()
    1e-06
    """
    global _EPSILON
    _EPSILON = eps


def epsilon():
    """Get a `epsilon` parameter.

    Returns
    -------
    int or float

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch('pixyz.utils._EPSILON', 1e-07):
    ...     epsilon()
    1e-07
    """
    return _EPSILON


def get_dict_values(dicts, keys, return_dict=False):
    """Get values from `dicts` specified by `keys`.

    When `return_dict` is True, return values are in dictionary format.

    Parameters
    ----------
    dicts : dict

    keys : list

    return_dict : bool

    Returns
    -------
    dict or list

    Examples
    --------
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b"])
    [2]
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b", "d"], True)
    {'b': 2}
    """
    new_dicts = dict((key, dicts[key]) for key in keys if key in list(dicts.keys()))
    if return_dict is False:
        return list(new_dicts.values())

    return new_dicts


def delete_dict_values(dicts, keys):
    """Delete values from `dicts` specified by `keys`.

    Parameters
    ----------
    dicts : dict

    keys : list

    Returns
    -------
    new_dicts : dict

    Examples
    --------
    >>> delete_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    {'a': 1, 'c': 3}
    """
    new_dicts = dict((key, value) for key, value in dicts.items() if key not in keys)
    return new_dicts


def detach_dict(dicts):
    """Detach all values in `dicts`.

    Parameters
    ----------
    dicts : dict

    Returns
    -------
    dict
    """
    return {k: v.detach() for k, v in dicts.items()}


def replace_dict_keys(dicts, replace_list_dict):
    """ Replace values in `dicts` according to `replace_list_dict`.

    Parameters
    ----------
    dicts : dict
        Dictionary.
    replace_list_dict : dict
        Dictionary.

    Returns
    -------
    replaced_dicts : dict
        Dictionary.

    Examples
    --------
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":"x","b":"y"})
    {'x': 1, 'y': 2, 'c': 3}
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":"x","e":"y"})  # keys of `replace_list_dict`
    {'x': 1, 'b': 2, 'c': 3}
    """
    replaced_dicts = dict([(replace_list_dict[key], value) if key in list(replace_list_dict.keys())
                           else (key, value) for key, value in dicts.items()])

    return replaced_dicts


def replace_dict_keys_split(dicts, replace_list_dict):
    """ Replace values in `dicts` according to :attr:`replace_list_dict`.

    Replaced dict is splitted by :attr:`replaced_dict` and :attr:`remain_dict`.

    Parameters
    ----------
    dicts : dict
        Dictionary.
    replace_list_dict : dict
        Dictionary.

    Returns
    -------
    replaced_dict : dict
        Dictionary.
    remain_dict : dict
        Dictionary.

    Examples
    --------
    >>> replace_list_dict = {'a': 'loc'}
    >>> x_dict = {'a': 0, 'b': 1}
    >>> print(replace_dict_keys_split(x_dict, replace_list_dict))
    ({'loc': 0}, {'b': 1})

    """
    replaced_dict = {replace_list_dict[key]: value for key, value in dicts.items()
                     if key in list(replace_list_dict.keys())}

    remain_dict = {key: value for key, value in dicts.items()
                   if key not in list(replace_list_dict.keys())}

    return replaced_dict, remain_dict


# immutable dict class
class FrozenSampleDict:
    def __init__(self, dict_):
        self.dict = dict_

    def __hash__(self):
        hashes = [(hash(key), hash(value)) for key, value in self.dict.items()]
        return hash(tuple(hashes))

    def __eq__(self, other):
        class EqTensor:
            def __init__(self, tensor):
                self.tensor = tensor

            def __eq__(self, other):
                if not torch.is_tensor(self.tensor):
                    return self.tensor == other.tensor
                return torch.all(self.tensor.eq(other.tensor))
        return {key: EqTensor(value) for key, value in self.dict.items()} ==\
               {key: EqTensor(value) for key, value in other.dict.items()}


def lru_cache_for_sample_dict(maxsize=0):
    """
    Memoize the calculation result linked to the argument of sample dict.
    Note that dictionary arguments of the target function must be sample dict.

    Parameters
    ----------
    maxsize: cache size prepared for the target method

    Returns
    -------
    decorator function

    Examples
    --------
    >>> import time
    >>> import torch.nn as nn
    >>> import pixyz.utils as utils
    >>> # utils.CACHE_SIZE = 2  # you can also use this module option to enable all memoization of distribution
    >>> import pixyz.distributions as pd
    >>> class LongEncoder(pd.Normal):
    ...     def __init__(self):
    ...         super().__init__(cond_var=['y'], var=['x'])
    ...         self.nn = nn.Sequential(*(nn.Linear(1,1) for i in range(10000)))
    ...     def forward(self, y):
    ...         return {'loc': self.nn(y), 'scale': torch.ones(1,1)}
    ...     @lru_cache_for_sample_dict(maxsize=2)
    ...     def get_params(self, params_dict={}, **kwargs):
    ...         return super().get_params(params_dict, **kwargs)
    >>> def measure_time(func):
    ...     start = time.time()
    ...     func()
    ...     elapsed_time = time.time() - start
    ...     return elapsed_time
    >>> le = LongEncoder()
    >>> y = torch.ones(1, 1)
    >>> t_sample1 = measure_time(lambda:le.sample({'y': y}))
    >>> print ("sample1:{0}".format(t_sample1) + "[sec]") # doctest: +SKIP
    >>> t_log_prob = measure_time(lambda:le.get_log_prob({'x': y, 'y': y}))
    >>> print ("log_prob:{0}".format(t_log_prob) + "[sec]") # doctest: +SKIP
    >>> t_sample2 = measure_time(lambda:le.sample({'y': y}))
    >>> print ("sample2:{0}".format(t_sample2) + "[sec]") # doctest: +SKIP
    >>> assert t_sample1 > t_sample2, "processing time increases: {0}".format(t_sample2 - t_sample1)
    """
    if not CACHE_SIZE and not maxsize:
        return lambda x: x
    if not maxsize:
        maxsize = CACHE_SIZE
    raw_decorating_function = functools.lru_cache(maxsize=maxsize, typed=False)

    def decorating_function(user_function):
        def wrapped_user_function(sender, *args, **kwargs):
            new_args = list(args)
            new_kwargs = dict(kwargs)
            for i in range(len(args)):
                if isinstance(args[i], FrozenSampleDict):
                    new_args[i] = args[i].dict
            for key in kwargs.keys():
                if isinstance(kwargs[key], FrozenSampleDict):
                    new_kwargs[key] = kwargs[key].dict
            return user_function(sender, *new_args, **new_kwargs)

        def frozen(wrapper):
            def frozen_wrapper(sender, *args, **kwargs):
                new_args = list(args)
                new_kwargs = dict(kwargs)
                for i in range(len(args)):
                    if isinstance(args[i], list):
                        new_args[i] = tuple(args[i])
                    elif isinstance(args[i], dict):
                        new_args[i] = FrozenSampleDict(args[i])
                for key in kwargs.keys():
                    if isinstance(kwargs[key], list):
                        new_kwargs[key] = tuple(kwargs[key])
                    elif isinstance(kwargs[key], dict):
                        new_kwargs[key] = FrozenSampleDict(kwargs[key])
                result = wrapper(sender, *new_args, **new_kwargs)
                return result
            return frozen_wrapper
        return frozen(raw_decorating_function(wrapped_user_function))
    return decorating_function


def tolist(a):
    """Convert a given input to the dictionary format.

    Parameters
    ----------
    a : list or other

    Returns
    -------
    list

    Examples
    --------
    >>> tolist(2)
    [2]
    >>> tolist([1, 2])
    [1, 2]
    >>> tolist([])
    []
    """
    if type(a) is list:
        return a
    return [a]


def sum_samples(samples):
    """Sum a given sample across the axes.

    Parameters
    ----------
    samples : torch.Tensor
        Input sample. The number of this axes is assumed to be 4 or less.

    Returns
    -------
    torch.Tensor
        Sum over all axes except the first axis.


    Examples
    --------
    >>> a = torch.ones([2])
    >>> sum_samples(a).size()
    torch.Size([2])
    >>> a = torch.ones([2, 3])
    >>> sum_samples(a).size()
    torch.Size([2])
    >>> a = torch.ones([2, 3, 4])
    >>> sum_samples(a).size()
    torch.Size([2])
    """

    dim = samples.dim()
    if dim == 1:
        return samples
    elif dim <= 4:
        dim_list = list(torch.arange(samples.dim()))
        samples = torch.sum(samples, dim=dim_list[1:])
        return samples
    raise ValueError("The number of sample axes must be any of 1, 2, 3, or 4, "
                     "got %s." % dim)


def print_latex(obj):
    """Print formulas in latex format.

    Parameters
    ----------
    obj : pixyz.distributions.distributions.Distribution, pixyz.losses.losses.Loss or pixyz.models.model.Model.

    """

    if isinstance(obj, pixyz.distributions.distributions.Distribution):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.distributions.distributions.DistGraph):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.losses.losses.Loss):
        latex_text = obj.loss_text
    elif isinstance(obj, pixyz.models.model.Model):
        latex_text = obj.loss_cls.loss_text

    return Math(latex_text)


def convert_latex_name(name):
    return sympy.latex(sympy.Symbol(name))


# def print_pgm(dist, filename, enc_dist=None):
#     """
#     plot graph of probablistic graphical model of Distribution.
# 
#     Parameters
#     ----------
#     dist: Distribution
#     filename: str
#     enc_dists: list of Distribution
# 
#     Examples
#     --------
#     >>> dist = _prepare_dist()
#     >>> print_pgm(dist, "tmp_daft.png")
#     """
#     default_node_size = 1
#     default_margin = 1
# 
#     # TODO: 描画用のグラフに整形する
#     glgraph, var_dict = _dist2graph(dist, default_node_size, forward=True)
#     # エンコーダ用の追加ノード追加エッジは後回し
#     # if enc_dist is not None:
#     #     _dist2graph(enc_dist, default_node_size, forward=False, glgraph=glgraph, var_dict=var_dict)
# 
#     # TODO: GPLでないアルゴリズムでグラフを配置する
#     # layout = _GeneralSugiyamaLayout(glgraph, default_node_size, default_margin)
#     # layout.draw()
#     pos = layout(dag)
# 
#     # TODO: 描画データを元に描画する
#     _draw_graph(glgraph, filename, default_node_size)


def layout(directed_acyclic_graph: nx.DiGraph):
    """
    
    Parameters
    ----------
    directed_acyclic_graph

    Returns
    -------

    Examples
    --------
    >>> dag = nx.DiGraph()
    >>> dag.add_edge(1, 2)
    >>> dag.add_edge(1, 3)
    >>> dag.add_edge(2, 3)
    >>> dag.add_edge(2, 4)
    >>> dag.add_edge(3, 4)
    >>> layout(dag)
    """
    dag = directed_acyclic_graph
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("only directed acyclic graph is supported.")
    margin = 1
    # 連結成分に分ける
    poss = []
    comps = nx.weakly_connected_components(dag)
    for comp in comps:
        # in edgeのみの「ground node」を見つける
        ground_nodes = []
        for node in comp:
            if len(dag.succ[node]) == 0:
                ground_nodes.append(node)
        # ground nodeからの生の高さを計算する
        height = {}
        max_height = 0
        for gn in ground_nodes:
            # 現在の高さを0とする
            height[gn] = 0
            # in edge先に現在の高さ+1より低いノードがあれば更新を続ける
            tracking_nodes = {gn}
            while len(tracking_nodes) != 0:
                tn = tracking_nodes.pop()
                for pn in dag.pred[tn]:
                    if pn not in height or height[pn] < height[tn] + 1:
                        height[pn] = height[tn] + 1
                        tracking_nodes.add(pn)
                        max_height = max(height[pn], max_height)
        # ノードごとにin edge, out edgeの集合内に同じ高さのものを見つけ，その数を記録する
        causal_block_dict = {}
        for node in comp:
            base_height = height[node]
            blocks = {}
            causal_block_dict[node] = []
            for pred in itertools.chain(dag.pred[node], dag.succ[node]):
                h = height[pred]
                if h not in blocks:
                    blocks[h] = []
                blocks[h].append(pred)
            causal_block_dict[node].extend((h, block, abs(h - base_height)) for h, block in blocks.items())
        causal_blocks = sorted(itertools.chain(*causal_block_dict.values()), key=lambda item: (item[2], -len(item[1])))

        # 同じ高さのまとまりが大きいものから，1次元配列として隣接するようにソートしてブロックとして保持する
        layered_orders = [Block() for _ in range(max_height + 1)]
        for node, h in height.items():
            layered_orders[h].append(node)
        for h, block, delta in causal_blocks:
            layered_orders[h].concat(block)

        # （オプション）out edgeが2つまでで，対象のノードが隣接している場合，その間にノードを挿入する
        # ノードを実際に配置する
        pos = {}
        h_wide_layer = np.argmax([len(layer) for layer in layered_orders]).item()
        # もっとも幅の広い層を等間隔に配置する
        for i, node in enumerate(layered_orders[h_wide_layer].nodes):
            pos[node] = (i, h_wide_layer)
        # もっとも幅の広い層の周りを順に配置する
        # def roundrobin(*iterables):
        #     return list(it.next() for it in itertools.cycle(map(iter, iterables)))
        # booked_layers = [(h, layered_orders[h]) for h in
        #                  roundrobin(range(h_wide_layer), range(max_height, h_wide_layer, -1))]
        booked_layers = ((h, layered_orders[h]) for h in
                         itertools.chain(range(h_wide_layer - 1, -1, -1), range(h_wide_layer + 1, max_height + 1)))
        for h, layer in booked_layers:
            # 最も一次接続の多いノードから配置する
            edge_counts = []
            for node in layer.nodes:
                count = 0
                for _, block, delta in causal_block_dict[node]:
                    if delta == 1:
                        count += len(block)
                edge_counts.append(count)
            i_dense_node = np.argmax(edge_counts).item()
            booked_node_indexes = itertools.chain(range(i_dense_node - 1, -1, -1), range(i_dense_node + 1, len(layer)))
            # 一次接続するノードの重心と最小間隔の兼ね合いで各層のノードを配置する
            pos[layer[i_dense_node]] = _locate_node(i_dense_node, layer, height, pos, causal_block_dict, margin=margin)
            for i in booked_node_indexes:
                pos[layer[i]] = _locate_node(i, layer, height, pos, causal_block_dict, margin=margin)
        # （オプション）skip connectionのエッジ曲線を生成する
        poss.append(pos)
    # 連結木を配置する
    global_pos = {}
    tree_center = np.array([0, 0])
    for pos in poss:
        tree_rect = [0, 0, 1, 1]
        for node, point in pos.items():
            tree_rect[0] = min(tree_rect[0], point[0] - margin)
            tree_rect[1] = min(tree_rect[1], point[1] - margin)
            tree_rect[2] = max(tree_rect[2], point[0] + margin)
            tree_rect[3] = max(tree_rect[3], point[1] + margin)
        for node, point in pos.items():
            global_pos[node] = np.array(pos[node]) + tree_center - np.array(tree_rect[:2])
        tree_center += np.array([tree_rect[2] - tree_rect[0], 0])
    return global_pos


def _locate_node(i: int, layer, height: dict, pos: dict, causal_block_dict: dict, margin=1):
    target_node = layer[i]
    connected_node_group = causal_block_dict[target_node]
    left_node = layer[i - 1] if i != 0 else None
    right_node = layer[i + 1] if i != len(layer) - 1 else None
    if left_node not in pos:
        left_node = None
    if right_node not in pos:
        right_node = None
    primary_groups = filter(lambda tup: tup[2] == 1 and tup[1][0] in pos, connected_node_group)
    def nodes(block):
        if isinstance(block, Block):
            return block.nodes
        else:
            return block
    mean_x = np.mean([np.mean([pos[node][0] for node in nodes(block)]) for h, block, delta in primary_groups])
    clipped_x = np.clip(mean_x, pos[left_node] + margin if left_node else -np.inf,
                        pos[right_node] - margin if right_node else np.inf)
    return clipped_x, height[target_node]


class Block:
    def __init__(self):
        # ブロック内にもブロックがある
        self.items = []
        self.left_lock = False
        self.right_lock = False

    def append(self, node):
        self.items.append(node)

    def contains(self, node):
        for item in self.items:
            if item == node:
                return True
            if isinstance(item, Block) and item.contains(node):
                return True
        return False

    @property
    def nodes(self):
        for item in self.items:
            if isinstance(item, Block):
                for node in item.nodes:
                    yield node
            else:
                yield item

    def __len__(self):
        return sum(1 for _ in self.nodes)

    def __getitem__(self, item):
        for i, node in enumerate(self.nodes):
            if item == i:
                return node
        raise IndexError()

    def _get_block(self, node):
        for item in self.items:
            if item == node:
                return item
            if isinstance(item, Block) and item.contains(node):
                return item
        return None

    def is_single_node(self, node):
        for item in self.items:
            if item == node:
                return True
            if isinstance(item, Block):
                if len(item.items) != 1:
                    return False
                if item.left_lock or item.right_lock:
                    return False
                return item.is_single_node(node)
        return False

    def is_side_node(self, node):
        for item in self.items:
            if item == node:
                return True
            if isinstance(item, Block) and item.contains(node):
                if item.left_lock and item.right_lock:
                    return False
                return item.is_side_node(node)
        return False

    def concat(self, block):
        if len(block) < 2:
            return
        # ソート可能かチェック
        side_nodes = []
        for node in block:
            if self.is_single_node(node):
                continue
            if self.is_side_node(node):
                side_nodes.append(node)
                if len(side_nodes) > 2:
                    return
        # ブロック単位でソート
        left_i = -1
        if len(side_nodes) > 0:
            left_block = self._get_block(side_nodes[0])
            left_block.move_side(side_nodes[0])
            if not left_block.right_lock:
                self._flip(left_block)
            left_i = self.items.index(left_block)
            left_block.right_lock = True

        if len(block) - len(side_nodes) > 0:
            center_block = Block()
            for node in block:
                if node not in side_nodes:
                    center_block.append(node)
                    old_block = self._get_block(node)
                    self.items.remove(old_block)
            self.items.insert(left_i + 1, center_block)
            if len(side_nodes) > 0:
                center_block.left_lock = True
            if len(side_nodes) > 1:
                center_block.right_lock = True

        if len(side_nodes) > 1:
            right_block = self._get_block(side_nodes[1])
            right_block.move_side(side_nodes[1])
            if not right_block.left_lock:
                self._flip(right_block)
            self._move(right_block, left_i + 2)
            right_block.left_lock = True

    def _get_locked_range(self, block):
        i_block = self.items.index(block)
        i_start, i_end = i_block, i_block + 1
        while True:
            block = self.items[i_start]
            if not isinstance(block, Block) or not block.left_lock:
                break
            i_start = i_start - 1
        while True:
            block = self.items[i_end - 1]
            if not isinstance(block, Block) or not block.right_lock:
                break
            i_end = i_end - 1
        return i_start, i_end

    def move_side(self, node):
        target_block = self._get_block(node)
        target_block.move_side(node)
        left_side = (target_block[0] == node)

        if not self.left_lock:
            if not left_side:
                self._flip(target_block)
            self._move(target_block, 0)
        elif not self.right_lock:
            if left_side:
                self._flip(target_block)
            self._move(target_block, -1)
        else:
            raise ValueError()

    def _flip(self, block):
        if not isinstance(block, Block):
            return
        i = self.items.index(block)
        left_block = self.items[i - 1] if block.left_lock else None
        right_block = self.items[i + 1] if block.right_lock else None
        block.reverse()
        if left_block and right_block:
            self.items[i - 1] = right_block
            self.items[i + 1] = left_block
        elif left_block:
            self.items.pop(i - 1)
            self.items.append(left_block)
        elif right_block:
            self.items.pop(i + 1)
            self.items.insert(i, right_block)

    def reverse(self):
        tmp = self.left_lock
        self.left_lock = self.right_lock
        self.right_lock = tmp
        self.items.reverse()
        for item in self.items:
            if isinstance(item, Block):
                item.reverse()

    def _move(self, block, dest):
        i_start, i_end = self._get_locked_range(block)
        moving_blocks = self.items[i_start:i_end]
        for _ in range(i_end - i_start):
            self.items.pop(i_start)
        for block in moving_blocks:
            self.items.insert(dest, block)


# def _dist2graph(dist, node_size, forward=True):#, glgraph: glg.Graph = None, var_dict=None):
#     var_dict = {}
#     glgraph = glg.Graph()
#     # if var_dict is None:
#     #     var_dict = {}
#     # if glgraph is None:
#     #     glgraph = glg.Graph()
#     if isinstance(dist, pixyz.distributions.distributions.MultiplyDistribution):
#         _dist2graph(dist._parent, node_size, forward, glgraph, var_dict)
#         _dist2graph(dist._child, node_size, forward, glgraph, var_dict)
#     elif isinstance(dist, pixyz.distributions.distributions.MarginalizeVarDistribution):
#         # marginalized var is not observed.
#         _dist2graph(dist.p, node_size, forward, glgraph, var_dict)
#         for var_name in dist._marginalize_list:
#             var_dict[var_name].data = _PGMNode(var_name, node_size, observed=False)
#     elif isinstance(dist, pixyz.distributions.distributions.ReplaceVarDistribution):
#         _dist2graph(dist.p, node_size, forward, glgraph, var_dict)
#         _replace_var_name(glgraph, var_dict, dist._replace_dict, forward)
#     else:
#         var = dist.var
#         for var_name in var:
#             if var_name not in var_dict:
#                 var_dict[var_name] = glg.Vertex(_PGMNode(var_name, node_size))
#             glgraph.add_vertex(var_dict[var_name])
#             for cond_var_name in dist.cond_var:
#                 if cond_var_name not in var_dict:
#                     var_dict[cond_var_name] = glg.Vertex(_PGMNode(cond_var_name, node_size))
#                 # the direction of the edge is inversed because root nodes are aligned with the baseline.
#                 to_node = var_dict[cond_var_name]
#                 from_node = var_dict[var_name]
#                 if not forward:
#                     from_node, to_node = (to_node, from_node)
#                 glgraph.add_edge(glg.Edge(from_node, to_node, data=_PGMEdge(dotted=not forward)))
#     return glgraph, var_dict
# 
# 
# def _replace_var_name(glgraph, var_dict, replace_pair_dict, forward=True):
#     for key, replaced_key in replace_pair_dict.items():
#         if key not in var_dict:
#             raise ValueError(f"replaced_pair_dict has a unknown source key {key}.")
#         if replaced_key in var_dict:
#             raise ValueError(f"replaced_pair_dict has a existing destination key {replaced_key}.")
# 
#         value = var_dict[key]
#         renamed_value = glg.Vertex(_PGMNode(replaced_key, value.data.node_size, observed=value.data.observed))
#         # replace edges
#         renamed_edges_in = [glg.Edge(edge.v[0] if forward else renamed_value,
#                                      renamed_value if forward else edge.v[0],
#                                      data=_PGMEdge(dotted=not forward)) for edge in value.e_in()]
#         renamed_edges_out = [glg.Edge(renamed_value if forward else edge.v[1],
#                                       edge.v[1] if forward else renamed_value,
#                                       data=_PGMEdge(dotted=not forward)) for edge in value.e_out()]
# 
#         var_dict[replaced_key] = renamed_value
#         glgraph.add_vertex(renamed_value)
#         for edge in renamed_edges_in + renamed_edges_out:
#             glgraph.add_edge(edge)
#         _glf_remove_vertex(glgraph, value)
#         del var_dict[key]
# 
# 
# # patch of bug? of grandalf
# def _glf_remove_vertex(graph: glg.Graph, vertex: glg.Vertex):
#     if len(vertex.c.sV) == 1:
#         graph.C.remove(vertex.c)
#     else:
#         graph.remove_vertex(vertex)
# 
# 
# # generalization of sugiyama layout for multiple components of DAG
# class _GeneralSugiyamaLayout:
#     def __init__(self, graph: glg.Graph, node_size, margin):
#         self.node_size = node_size
#         self.margin = margin
# 
#         class defaultview(object):
#             w, h = (self.node_size, self.node_size)
# 
#         for v in graph.V():
#             v.view = defaultview()
#         self.component_layouts = [gly.SugiyamaLayout(comp) for comp in graph.C]
#         for c_layout in self.component_layouts:
#             c_layout.dw, c_layout.dh = (self.node_size, self.node_size)
#             c_layout.xspace, c_layout.yspace = (self.margin, self.margin)
# 
#     def draw(self):
#         top_left = [0, 0]
#         for c_layout in self.component_layouts:
#             c_layout.init_all()
#             c_layout.draw()
# 
#             # arrange components so that they do not overlap
#             xyrange = _get_vertex_range(c_layout.g.sV)
#             for v in c_layout.g.sV:
#                 v.view.xy = (v.view.xy[0] + top_left[0] - xyrange[0], v.view.xy[1] + top_left[1] - xyrange[2])
#             top_left = [top_left[0] + xyrange[1] - xyrange[0] + self.margin, top_left[1]]
# 
# 
# def _glf_edge_count(graph):
#     from_to_dict = {}
#     for e in graph.E():
#         if e.v in from_to_dict:
#             from_to_dict[e.v].append(e)
#         elif (e.v[1], e.v[0]) in from_to_dict:
#             from_to_dict[(e.v[1], e.v[0])].append(e)
#         else:
#             from_to_dict[e.v] = [e]
#     result = {}
#     for e in graph.E():
#         if e.v in from_to_dict:
#             result[e] = len(from_to_dict[e.v])
#         elif (e.v[1], e.v[0]) in from_to_dict:
#             result[e] = len(from_to_dict[(e.v[1], e.v[0])])
#         else:
#             raise ValueError()
#     return result
# 
# 
# def _draw_graph(graph, filename, node_size, transparent=False):
#     fig = plt.figure(frameon=not transparent)
#     xyrange = _get_vertex_range(graph.V())
#     ax = fig.add_axes([0, 0, 1, 1], aspect=1.)
#     linewidth = node_size * 4
#     pyplot_margin = linewidth / 40
#     ax.set_xlim(xyrange[0] - pyplot_margin, xyrange[1] + pyplot_margin)
#     ax.set_ylim(xyrange[2] - pyplot_margin, xyrange[3] + pyplot_margin)
#     # ax.axis('off')
# 
#     ec = _glf_edge_count(graph)
# 
#     for v in graph.V():
#         v.data.draw(v, ax, fig)
# 
#     for e in graph.E():
#         e.data.straight = (ec[e] == 1)
#         e.data.draw(e, ax, fig, node_size)
# 
#     ax.set_facecolor('w')
#     with open(filename, 'wb') as f:
#         fig.canvas.print_png(f)
# 
# 
# def _convert_data2point_scale(ax, fig, target):
#     # constant value of matplotlib
#     ppi = 72
#     # ax_length = ax.bbox.get_points()[1][0] -ax.bbox.get_points()[0][0]
#     # ax_point = ax_length*ppi/fig.dpi
#     # result = ax_point/x_size
#     # TODO: スケールが謎
#     tf = ax.transData.transform([1, 0]) * (ppi / fig.dpi)
#     return target * (tf[0] - tf[1])
# 
# 
# def _convert_point2data_scale(ax, fig, target):
#     # constant value of matplotlib
#     ppi = 72
#     # ax_length = ax.bbox.get_points()[1][0] -ax.bbox.get_points()[0][0]
#     # ax_point = ax_length*ppi/fig.dpi
#     # result = ax_point/x_size
#     return ax.transData.invereted().transform(target) / (ppi / fig.dpi)
# 
# 
# class _PGMNode:
#     def __init__(self, var_name, node_size, observed=True):
#         self.var_name = var_name
#         self.observed = observed
#         self.node_size = node_size
#         # magic scaling number for matplotlib.patches
#         self.radius = node_size / 2
#         self.linewidth = node_size * 2
#         self.fontsize = node_size * 30
# 
#     def draw(self, g_vertex, ax, fig):
#         fc = 'gray' if self.observed else 'w'
#         circle = patches.Circle(xy=g_vertex.view.xy, radius=self.radius, fc=fc, ec='k', linewidth=self.linewidth)
#         ax.add_patch(circle)
#         ax.text(g_vertex.view.xy[0], g_vertex.view.xy[1], rf"${self.var_name}$", fontsize=self.fontsize,
#                 verticalalignment='center', horizontalalignment='center')
# 
# 
# class _PGMEdge:
#     def __init__(self, straight=False, dotted=True):
#         self.linestyle = '-' if not dotted else '--'
#         self.straight = straight
#         self.forward = not dotted
# 
#     def draw(self, glf_edge, ax, fig, node_size):
#         # magic scaling number for matplotlib.patches
#         linewidth = node_size * 2
#         # path = patches.Path([e.v[0].view.xy, e.v[1].view.xy])
#         node0, node1 = glf_edge.v
#         if not self.forward:
#             node0, node1 = (node1, node0)
#         self.add_arrow(ax, node0.view.xy, node1.view.xy, _convert_data2point_scale(ax, fig, node_size/2),
#                        linewidth, self.linestyle,
#                        node_size * 0.3, node_size * 1.3, 0 if self.straight else 0.3)
# 
#     @staticmethod
#     def add_arrow(ax, v1, v2, shrink, linewidth, linestyle, head_width, head_length, rad, path=None):
#         # ax.add_patch(patches.FancyArrowPatch(v1, v2,
#         #                                      arrowstyle=patches.ArrowStyle("-|>", head_length=head_length,
#         #                                                                    head_width=head_width),
#         #                                      arrow_transmuter=None,
#         #                                      connectionstyle=patches.ConnectionStyle("arc3", rad=rad), connector=None,
#         #                                      patchA=None, patchB=None, shrinkA=shrink, shrinkB=shrink,
#         #                                      mutation_scale=1, mutation_aspect=None, dpi_cor=1,
#         #                                      color='k', path=path, capstyle='butt',
#         #                                      linewidth=linewidth, linestyle=linestyle,
#         #                                      ))
#         ax.annotate('', xy=v1, xycoords='data', xytext=v2, textcoords='data',
#                     arrowprops=dict(arrowstyle=
#                                     patches.ArrowStyle("-|>", head_length=head_length, head_width=head_width),
#                                     color='k', path=path,
#                                     shrinkA=shrink, shrinkB=shrink,
#                                     linewidth=linewidth, linestyle=linestyle,
#                                     capstyle='butt',
#                                     connectionstyle=patches.ConnectionStyle("arc3", rad=rad)))
# 
# 
# def _get_vertex_range(vertices):
#     xyrange = [np.inf, -np.inf, np.inf, -np.inf]
#     for v in vertices:
#         xyrange[0] = min(xyrange[0], v.view.xy[0] - v.view.w / 2)
#         xyrange[1] = max(xyrange[1], v.view.xy[0] + v.view.w / 2)
#         xyrange[2] = min(xyrange[2], v.view.xy[1] - v.view.h / 2)
#         xyrange[3] = max(xyrange[3], v.view.xy[1] + v.view.h / 2)
#     return xyrange


def _prepare_dist():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pixyz.distributions.exponential_distributions import Normal
    from pixyz.distributions.distributions import ReplaceVarDistribution, MarginalizeVarDistribution
    x_dim = 20
    y_dim = 30
    a_dim = 50
    long_name = r"\phi_{x,y}"

    # b -> a
    class P0(Normal):
        def __init__(self):
            super(P0, self).__init__(cond_var=["b"], var=["a"], name="p_{0}")

            self.fc21 = nn.Linear(10, a_dim)
            self.fc22 = nn.Linear(10, a_dim)

        def forward(self, b):
            return {"loc": self.fc21(b), "scale": F.softplus(self.fc22(b))}

    # (y,a) -> x=phi
    class P1(Normal):
        def __init__(self):
            super(P1, self).__init__(cond_var=["y", "a"], var=["x"], name="p_{1}")

            self.fc1 = nn.Linear(y_dim, 10)
            self.fc2 = nn.Linear(a_dim, 10)
            self.fc21 = nn.Linear(10 + 10, 20)
            self.fc22 = nn.Linear(10 + 10, 20)

        def forward(self, a, y):
            h1 = F.relu(self.fc1(y))
            h2 = F.relu(self.fc2(a))
            h12 = torch.cat([h1, h2], 1)
            return {"loc": self.fc21(h12), "scale": F.softplus(self.fc22(h12))}

    # (phi, y) -> z
    class P2(Normal):
        def __init__(self):
            super(P2, self).__init__(cond_var=[long_name, "y"], var=["z"], name="p_{2}")

            self.fc3 = nn.Linear(x_dim, 30)
            self.fc4 = nn.Linear(30 + y_dim, 400)
            self.fc51 = nn.Linear(400, 20)
            self.fc52 = nn.Linear(400, 20)

        def forward(self, x, y):
            h3 = F.relu(self.fc3(x))
            h4 = F.relu(self.fc4(torch.cat([h3, y], 1)))
            return {"loc": self.fc51(h4), "scale": F.softplus(self.fc52(h4))}

    # z -> y
    class Q1(Normal):
        def __init__(self):
            super(Q1, self).__init__(cond_var=["z"], var=["y"], name="q_{1}")

            self.fc1 = nn.Linear(20, 10)
            self.fc21 = nn.Linear(10, y_dim)
            self.fc22 = nn.Linear(10, y_dim)

        def forward(self, a, z):
            h1 = F.relu(self.fc1(z))
            return {"loc": self.fc21(h1), "scale": F.softplus(self.fc22(h1))}

    p4 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["a"], features_shape=[a_dim], name="p_{4}")
    p6 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["y"], features_shape=[y_dim], name="p_{6}")
    p7 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["w"], features_shape=[y_dim], name="p_{7}")
    p8 = ReplaceVarDistribution(p7, {'w': 'W'})

    p0 = P0()
    p1 = P1()
    p1_r = ReplaceVarDistribution(p1, {'x': long_name})
    p2 = P2()
    p3 = p2 * p1_r
    p3.name = "p_{3}"
    p5 = p3 * p4
    p5.name = "p_{5}"
    p_all = p1_r * p2 * p4 * p6 * p8
    p_all2 = p0 * p1_r * p2 * p6 * p8
    p_all.name = "p_{all}"
    p_marg = MarginalizeVarDistribution(p_all, ['z'])
    p_marg2 = MarginalizeVarDistribution(p_all2, ['z'])
    q1 = Q1()
    return p_marg, q1, p_marg2


# def _test_print_pgm():
#     dist, enc_dist, dist2 = _prepare_dist()
#     print_pgm(dist, "tmp_daft.png", enc_dist=enc_dist)
#     print_pgm(dist2, "tmp_daft2.png", enc_dist=enc_dist)


if __name__ == "__main__":
    # _test_print_pgm()
    dag = nx.DiGraph()
    dag.add_edge(1, 2)
    dag.add_edge(1, 3)
    dag.add_edge(2, 3)
    dag.add_edge(2, 4)
    dag.add_edge(3, 4)
    # layout(dag)
    nx.draw_networkx(dag)
