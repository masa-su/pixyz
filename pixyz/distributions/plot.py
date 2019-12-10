import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import grandalf as glf
import grandalf.layouts as gly
import grandalf.graphs as glg
from grandalf.graphs import Vertex, Edge, Graph
import daft
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pixyz.distributions.distributions import Distribution, MultiplyDistribution, MarginalizeVarDistribution, ReplaceVarDistribution
from pixyz.distributions.exponential_distributions import Normal

rc("font", family="serif", size=12)
rc("text", usetex=True)


USE_DAFT = False


def test_plot():
    dist = prepare_dist()
    plot(dist, "tmp_daft.png")


def prepare_dist():
    x_dim = 20
    y_dim = 30
    z_dim = 40
    a_dim = 50
    batch_n = 2
    long_name = "too_long_name"

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

    p4 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["a"], features_shape=[a_dim], name="p_{4}")
    p6 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["y"], features_shape=[y_dim], name="p_{6}")
    p7 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["w"], features_shape=[y_dim], name="p_{7}")
    p8 = ReplaceVarDistribution(p7, {'w': 'W'})

    x = torch.from_numpy(np.random.random((batch_n, x_dim)).astype("float32"))
    y = torch.from_numpy(np.random.random((batch_n, y_dim)).astype("float32"))
    a = torch.from_numpy(np.random.random((batch_n, a_dim)).astype("float32"))

    p1 = P1()
    p1_r = ReplaceVarDistribution(p1, {'x': long_name})
    p2 = P2()
    p3 = p2 * p1_r
    p3.name = "p_{3}"
    p5 = p3 * p4
    p5.name = "p_{5}"
    p_all = p1_r * p2 * p4 * p6 * p8
    p_all.name = "p_{all}"
    p_marg = MarginalizeVarDistribution(p_all, ['z'])
    return p_marg


def get_vertex_range(vertices):
    xyrange = [np.inf, -np.inf, np.inf, -np.inf]
    for v in vertices:
        xyrange[0] = min(xyrange[0], v.view.xy[0] - v.view.w / 2)
        xyrange[1] = max(xyrange[1], v.view.xy[0] + v.view.w / 2)
        xyrange[2] = min(xyrange[2], v.view.xy[1] - v.view.h / 2)
        xyrange[3] = max(xyrange[3], v.view.xy[1] + v.view.h / 2)
    return xyrange


# 複数の連結成分へのSugiyamaLayoutの一般化
class GeneralSugiyamaLayout:
    def __init__(self, graph: glg.Graph, node_size, margin):
        self.node_size = node_size
        self.margin = margin

        class defaultview(object):
            w, h = (self.node_size, self.node_size)

        for v in graph.V():
            v.view = defaultview()
        self.component_layouts = [glf.layouts.SugiyamaLayout(comp) for comp in graph.C]
        for c_layout in self.component_layouts:
            c_layout.dw, c_layout.dh = (self.node_size, self.node_size)
            c_layout.xspace, c_layout.yspace = (self.margin, self.margin)

    def draw(self):
        top_left = [0, 0]
        for c_layout in self.component_layouts:
            c_layout.init_all()
            c_layout.draw()

            # コンポーネントが重ならないように並べる
            xyrange = get_vertex_range(c_layout.g.sV)
            for v in c_layout.g.sV:
                v.view.xy = (v.view.xy[0] + top_left[0] - xyrange[0], v.view.xy[1] + top_left[1] - xyrange[2])
            # # TODO: daftの謎スケールのためのヒューリスティクス
            # top_left = [top_left[0] + xyrange[1] - xyrange[0], top_left[1]]
            top_left = [top_left[0] + xyrange[1] - xyrange[0] + self.margin, top_left[1]]


def plot(dist, filename):
    default_node_size = 1
    # # TODO: daftの謎スケールのためのヒューリスティクス
    # default_margin = 0.1
    default_margin = 1
    glgraph = glg.Graph()
    make_dag_from_dist(dist, glgraph)
    layout = GeneralSugiyamaLayout(glgraph, default_node_size, default_margin)
    layout.draw()

    if USE_DAFT:
        draw_graph_by_daft(glgraph, filename, default_node_size)
    else:
        draw_graph_by_pyplot(glgraph, filename, default_node_size)


class PGMNode:
    def __init__(self, var_name, observed=True):
        self.var_name = var_name
        self.observed = observed


# grandalfにバグがあったので修正
def glf_remove_vertex(graph: glg.Graph, vertex: glg.Vertex):
    if len(vertex.c.sV) == 1:
        graph.C.remove(vertex.c)
    else:
        graph.remove_vertex(vertex)


def replace_var_name(glgraph, var_dict, replace_pair_dict):
    for key, replaced_key in replace_pair_dict.items():
        if key not in var_dict:
            raise ValueError(f"replaced_pair_dict has a unknown source key {key}.")
        if replaced_key in var_dict:
            raise ValueError(f"replaced_pair_dict has a existing destination key {replaced_key}.")

        value = var_dict[key]
        renamed_value = glg.Vertex(PGMNode(replaced_key, observed=value.data.observed))
        # エッジも保存して繋ぎ変える
        renamed_edges_in = [glg.Edge(edge.v[0], renamed_value) for edge in value.e_in()]
        renamed_edges_out = [glg.Edge(renamed_value, edge.v[1]) for edge in value.e_out()]
        var_dict[replaced_key] = renamed_value
        glgraph.add_vertex(renamed_value)
        for edge in renamed_edges_in + renamed_edges_out:
            glgraph.add_edge(edge)
        glf_remove_vertex(glgraph, value)
        del var_dict[key]


def make_dag_from_dist(dist: Distribution, glgraph: glg.Graph, var_dict=None):
    if var_dict is None:
        var_dict = {}
    if isinstance(dist, MultiplyDistribution):
        make_dag_from_dist(dist._parent, glgraph, var_dict)
        make_dag_from_dist(dist._child, glgraph, var_dict)
    elif isinstance(dist, MarginalizeVarDistribution):
        # 周辺化された変数を別表記する
        make_dag_from_dist(dist.p, glgraph, var_dict)
        for var_name in dist._marginalize_list:
            var_dict[var_name].data = PGMNode(var_name, observed=False)
    elif isinstance(dist, ReplaceVarDistribution):
        make_dag_from_dist(dist.p, glgraph, var_dict)
        replace_var_name(glgraph, var_dict, dist._replace_dict)
    else:
        var = dist.var
        for var_name in var:
            if var_name not in var_dict:
                var_dict[var_name] = glg.Vertex(PGMNode(var_name))
            glgraph.add_vertex(var_dict[var_name])
            for cond_var_name in dist.cond_var:
                if cond_var_name not in var_dict:
                    var_dict[cond_var_name] = glg.Vertex(PGMNode(cond_var_name))
                # エッジは逆向きに指定する
                glgraph.add_edge(glg.Edge(var_dict[cond_var_name], var_dict[var_name]))


def add_arrow(ax, v1, v2, linewidth, shrink, path=None):
    # arrowstyles=['-', '->', '-[', '-|>', '<-', '<->', '<|-', '<|-|>', ']-', ']-[', 'fancy', 'simple', 'wedge', '|-|']
    ax.annotate('', xy=v1, xycoords='data', xytext=v2, textcoords='data',
                arrowprops=dict(arrowstyle='-|>', color='k',
                                path=path,
                                shrinkA=shrink, shrinkB=shrink,
                                # patchA=patchA, patchB=patchB,
                                linewidth=linewidth,
                                linestyle='-.',
                                connectionstyle="arc3,rad=0.3"))


def get_edge_vertices(node0, node1, radius):
    start = np.array(node0.view.xy)
    end = np.array(node1.view.xy)
    # direction = end - start
    # v0 = start + radius * direction / np.linalg.norm(direction)
    # v1 = end - radius * direction / np.linalg.norm(direction)
    return start, end


def add_edge(ax, e, node_size):
    linewidth = node_size*4
    # path = patches.Path([e.v[0].view.xy, e.v[1].view.xy])
    node0, node1 = e.v
    v0, v1 = get_edge_vertices(node0, node1, node_size/2)
    add_arrow(ax, v0, v1, linewidth, node_size*33)


def draw_graph_by_daft(graph, filename, node_size):
    pgm = glf2daft(graph, node_size)
    pgm.render()
    pgm.savefig(filename)


def draw_graph_by_pyplot(graph, filename, node_size, transparent=False):
    fig = plt.figure(frameon=not transparent)
    xyrange = get_vertex_range(graph.V())
    ax = fig.add_axes([0, 0, 1, 1], aspect=1.)
    linewidth = node_size*4
    pyplot_margin = linewidth/40
    ax.set_xlim(xyrange[0] - pyplot_margin, xyrange[1] + pyplot_margin)
    ax.set_ylim(xyrange[2] - pyplot_margin, xyrange[3] + pyplot_margin)
    ax.axis('off')

    for e in graph.E():
        add_edge(ax, e, node_size)

    circles = {}
    for v in graph.V():
        fc = 'gray' if v.data.observed else 'w'
        circles[v] = patches.Circle(xy=v.view.xy, radius=node_size/2, fc=fc, ec='k', linewidth=linewidth)
        ax.add_patch(circles[v])
        ax.text(v.view.xy[0], v.view.xy[1], rf"${v.data.var_name}$", fontsize=node_size*30,
                verticalalignment='center', horizontalalignment='center')

    ax.set_facecolor('w')
    with open(filename, 'wb') as f:
        fig.canvas.print_png(f)


def glf2daft(graph, node_size):
    pgm = daft.PGM(node_unit=node_size, dpi=150)
    for v in graph.V():
        pgm.add_node(v.data.var_name, content=rf"${v.data.var_name}$", x=v.view.xy[0], y=v.view.xy[1],
                     observed=v.data.observed, fixed=False, shape="ellipse")

    for e in graph.E():
        pgm.add_edge(e.v[1].data.var_name, e.v[0].data.var_name, plot_params=dict(linestyle='-.'))
    return pgm


def test_output():
    g = prepare_graph()
    pgm = daft.PGM()
    for v in g.C[0].sV:
        pgm.add_node(str(v.data), rf"$x_{{{v.data}}}$", x=v.view.xy[0] / 30., y=-v.view.xy[1] / 30.)

    for e in g.C[0].sE:
        pgm.add_edge(str(e.v[0].data), str(e.v[1].data))

    pgm.render()
    pgm.savefig("tmp_daft.png", dpi=150)


def prepare_graph():
    V = [Vertex(data) for data in range(11)]
    X = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 0), (1, 4), (4, 5),
         (5, 6), (3, 6), (3, 7), (6, 8), (7, 8), (8, 9), (5, 9), (10, 6)]
    E = [Edge(V[v], V[w]) for (v, w) in X]
    g = Graph(V, E)
    g.add_edge(Edge(V[9], Vertex(10)))
    g.remove_edge(V[5].e_to(V[9]))
    g.remove_vertex(V[8])
    g.remove_edge(V[4].e_to(V[0]))

    class defaultview(object):
        w, h = 10, 10

    for v in V:
        v.view = defaultview()
    sug = gly.SugiyamaLayout(g.C[0])
    sug.init_all()
    sug.draw()
    return g


if __name__ == "__main__":
    test_plot()
