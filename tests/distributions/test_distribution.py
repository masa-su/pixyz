import pytest
from os.path import join as pjoin
import torch
from pixyz.distributions import Normal, MixtureModel, Categorical, FactorizedBernoulli
from pixyz.utils import lru_cache_for_sample_dict
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
import torch.nn.functional as F
from torch.optim import Adam
import pixyz.distributions as pd


class TestGraph:
    def test_rename_atomdist(self):
        normal = Normal(var=['x'], name='p')
        graph = normal.graph
        assert graph.name == 'p'
        normal.name = 'q'
        assert graph.name == 'q'

    def test_print(self):
        normal = Normal(var=['x'], name='p')
        print(normal.graph)

    def test_set_option(self):
        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1)
        dist.graph.set_option(dict(batch_n=4, sample_shape=(2, 3)), ['y'])
        sample = dist.sample()
        assert sample['y'].shape == torch.Size([2, 3, 4])
        assert sample['x'].shape == torch.Size([2, 3, 4])
        dist.graph.set_option({}, ['y'])
        assert dist.get_log_prob(sample,
                                 sum_features=True, feature_dims=None).shape == torch.Size([2])
        assert dist.get_log_prob(sample,
                                 sum_features=False).shape == torch.Size([2, 3, 4])

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * FactorizedBernoulli(
            var=['y'], probs=torch.tensor([0.3, 0.8]))
        dist.graph.set_option(dict(batch_n=3, sample_shape=(4,)), ['y'])
        sample = dist.sample()
        assert sample['y'].shape == torch.Size([4, 3, 2])
        assert sample['x'].shape == torch.Size([4, 3, 2])
        dist.graph.set_option(dict(), ['y'])
        assert dist.get_log_prob(sample, sum_features=True, feature_dims=[-1]).shape == torch.Size([4, 3])

    def test_sample_mean(self):
        dist = Normal(var=['x'], loc=0, scale=1) * Normal(var=['y'], cond_var=['x'], loc='x', scale=1)
        assert dist.sample(sample_mean=True)['y'] == torch.zeros(1)

    def test_input_extra_var(self):
        normal = Normal(var=['x'], loc=0, scale=1) * Normal(var=['y'], loc=0, scale=1)
        assert set(normal.sample({'z': torch.zeros(1)})) == set(('x', 'y', 'z'))
        assert normal.get_log_prob({'y': torch.zeros(1), 'x': torch.zeros(1),
                                    'z': torch.zeros(1)}).shape == torch.Size([1])
        assert set(normal.sample({'x': torch.zeros(1)})) == set(('x', 'y'))


class TestDistributionBase:
    def test_init_with_scalar_params(self):
        normal = Normal(loc=0, scale=1, features_shape=[2])
        assert normal.sample()['x'].shape == torch.Size([1, 2])
        assert normal.features_shape == torch.Size([2])

        normal = Normal(loc=0, scale=1)
        assert normal.sample()['x'].shape == torch.Size([1])
        assert normal.features_shape == torch.Size([])

    def test_batch_n(self):
        normal = Normal(loc=0, scale=1)
        assert normal.sample(batch_n=3)['x'].shape == torch.Size([3])

    def test_input_extra_var(self):
        normal = Normal(loc=0, scale=1)
        assert set(normal.sample({'y': torch.zeros(1)})) == set(('x', 'y'))
        assert normal.get_log_prob({'y': torch.zeros(1), 'x': torch.zeros(1)}).shape == torch.Size([1])
        assert set(normal.sample({'x': torch.zeros(1)})) == set(('x'))

    def test_sample_mean(self):
        dist = Normal(loc=0, scale=1)
        assert dist.sample(sample_mean=True)['x'] == torch.zeros(1)

    @pytest.mark.parametrize(
        "dist", [
            Normal(loc=0, scale=1),
            Normal(var=['x'], loc=0, scale=1) * Normal(var=['y'], loc=0, scale=1),
            # Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1),
        ],
    )
    def test_get_log_prob_feature_dims(self, dist):
        assert dist.get_log_prob(dist.sample(batch_n=4, sample_shape=(2, 3)),
                                 sum_features=True, feature_dims=None).shape == torch.Size([2])
        assert dist.get_log_prob(dist.sample(batch_n=4, sample_shape=(2, 3)),
                                 sum_features=True, feature_dims=[-2]).shape == torch.Size([2, 4])
        assert dist.get_log_prob(dist.sample(batch_n=4, sample_shape=(2, 3)),
                                 sum_features=True, feature_dims=[0, 1]).shape == torch.Size([4])
        assert dist.get_log_prob(dist.sample(batch_n=4, sample_shape=(2, 3)),
                                 sum_features=True, feature_dims=[]).shape == torch.Size([2, 3, 4])

    def test_get_log_prob_feature_dims2(self):
        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1)
        dist.graph.set_option(dict(batch_n=4, sample_shape=(2, 3)), ['y'])
        sample = dist.sample()
        assert sample['y'].shape == torch.Size([2, 3, 4])
        list(dist.graph._factors_from_variable('y'))[0].option = {}
        assert dist.get_log_prob(sample,
                                 sum_features=True, feature_dims=None).shape == torch.Size([2])
        assert dist.get_log_prob(sample,
                                 sum_features=True, feature_dims=[-2]).shape == torch.Size([2, 4])
        assert dist.get_log_prob(sample,
                                 sum_features=True, feature_dims=[0, 1]).shape == torch.Size([4])
        assert dist.get_log_prob(sample,
                                 sum_features=True, feature_dims=[]).shape == torch.Size([2, 3, 4])

    @pytest.mark.parametrize(
        "dist", [
            Normal(loc=0, scale=1),
            Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1),
        ])
    def test_unknown_option(self, dist):
        x_dict = dist.sample(unknown_opt=None)
        dist.get_log_prob(x_dict, unknown_opt=None)


class TestReplaceVarDistribution:
    def test_get_params(self):
        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1)
        result = dist.get_params({'y': torch.ones(1)})
        assert list(result.keys()) == ['loc', 'scale']

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z')
        result = dist.get_params({'z': torch.ones(1)})
        assert list(result.keys()) == ['loc', 'scale']

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z')
        with pytest.raises(ValueError):
            dist.get_params({'y': torch.ones(1)})

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(x='z')
        result = dist.get_params({'y': torch.ones(1)})
        assert list(result.keys()) == ['loc', 'scale']

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * Normal(var=['y'], loc=0, scale=1)
        with pytest.raises(NotImplementedError):
            dist.get_params()

    def test_sample_mean(self):
        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1)
        result = dist.sample_mean({'y': torch.ones(1)})
        assert result == torch.ones(1)

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z')
        result = dist.sample_mean({'z': torch.ones(1)})
        assert result == torch.ones(1)

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z')
        with pytest.raises(ValueError):
            dist.sample_mean({'y': torch.ones(1)})

    def test_sample_variance(self):
        dist = Normal(var=['x'], cond_var=['y'], loc=2, scale='y')
        result = dist.sample_variance({'y': torch.ones(1)})
        assert result == torch.ones(1)

        dist = Normal(var=['x'], cond_var=['y'], loc=2, scale='y').replace_var(y='z')
        result = dist.sample_variance({'z': torch.ones(1)})
        assert result == torch.ones(1)

        dist = Normal(var=['x'], cond_var=['y'], loc=2, scale='y').replace_var(y='z')
        with pytest.raises(ValueError):
            dist.sample_variance({'y': torch.ones(1)})

    def test_get_entropy(self):
        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1)
        truth = dist.get_entropy({'y': torch.ones(1)})

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z', x='y')
        result = dist.get_entropy({'z': torch.ones(1)})
        assert result == truth

        dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1).replace_var(y='z')
        with pytest.raises(ValueError):
            dist.get_entropy({'y': torch.ones(1)})


class TestMixtureDistribution:
    def test_sample_mean(self):
        dist = MixtureModel([Normal(loc=0, scale=1), Normal(loc=1, scale=1)], Categorical(probs=torch.tensor([1., 2.])))
        assert dist.sample(sample_mean=True)['x'] == torch.ones(1)


def test_memoization():
    exec_order = []

    class Encoder(Normal):
        def __init__(self, exec_order):
            super().__init__(var=["z"], cond_var=["x"], name="q")
            self.linear = torch.nn.Linear(10, 10)
            self.exec_order = exec_order

        @lru_cache_for_sample_dict()
        def get_params(self, params_dict={}, **kwargs):
            return super().get_params(params_dict, **kwargs)

        def forward(self, x):
            exec_order.append("E")
            return {"loc": self.linear(x), "scale": 1.0}

    class Decoder(Normal):
        def __init__(self, exec_order):
            super().__init__(var=["x"], cond_var=["z"], name="p")
            self.exec_order = exec_order

        @lru_cache_for_sample_dict()
        def get_params(self, params_dict={}, **kwargs):
            return super().get_params(params_dict, **kwargs)

        def forward(self, z):
            self.exec_order.append("D")
            return {"loc": z, "scale": 1.0}

    def prior():
        return Normal(var=["z"], name="p_{prior}", features_shape=[10], loc=torch.tensor(0.), scale=torch.tensor(1.))

    q = Encoder(exec_order)
    p = Decoder(exec_order)

    prior = prior()
    kl = KullbackLeibler(q, prior)

    mdl = VAE(q, p, regularizer=kl, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3})

    x = torch.zeros((10, 10))
    mdl.train({"x": x})
    assert exec_order == ["E", "D"]


# 重みの更新を挟んでキャッシュが使用されると、リセットなしでbackwardを使用したとpytorchからエラーが出る
# 重みの更新が終わったらキャッシュをclearしたほうがいいかもしれない
def test_memoization_for_prior():
    class GeneratorC0(pd.Categorical):
        def __init__(self, n_mix):
            super().__init__(var=["i"], name="p_i")
            self.pi = torch.nn.Parameter(torch.ones(n_mix), requires_grad=True)

        def forward(self):
            return {"probs": F.softplus(self.pi)}

    class GeneratorC1(pd.Categorical):
        def __init__(self, n_mix):
            super().__init__(var=["i"], cond_var=["c"], name="p_i")
            self.pi = torch.nn.Parameter(torch.ones(n_mix), requires_grad=True)

        def forward(self, c):
            return {"probs": F.softplus(self.pi)}

    x_dict = {'i': torch.Tensor([0, 1])}
    gi = GeneratorC0(2)
    optim = Adam(gi.parameters())
    optim.zero_grad()
    loss = gi.log_prob().eval(x_dict)
    loss.backward()
    optim.step()
    optim.zero_grad()
    loss = gi.log_prob().eval(x_dict)
    loss.backward()
    # error
    optim.zero_grad()
    loss = gi.get_log_prob(x_dict)
    loss.backward()
    # error

    # x_dict1 = {'i': torch.Tensor([0, 1]), 'c': torch.tensor(1)}
    # gi = GeneratorC1(2)
    # optim = Adam(gi.parameters())
    # optim.zero_grad()
    # loss = gi.log_prob().eval(x_dict1)
    # loss.backward()
    # optim.step()
    # optim.zero_grad()
    # loss = gi.log_prob().eval(x_dict1)
    # loss.backward()
    # # error
    # optim.zero_grad()
    # loss = gi.get_log_prob(x_dict)
    # loss.backward()
    # # error


@pytest.mark.parametrize(
    "no_contiguous_tensor", [
        torch.zeros(2, 3),
        torch.zeros(2, 3).T,
        torch.zeros(1).expand(3),
    ]
)
def test_save_dist(tmpdir, no_contiguous_tensor):
    # pull request:#110
    ones = torch.ones_like(no_contiguous_tensor)
    p = Normal(loc=no_contiguous_tensor, scale=ones)
    save_path = pjoin(tmpdir, "tmp.pt")
    torch.save(p.state_dict(), save_path)
    q = Normal(loc=ones, scale=3 * ones)
    assert not torch.all(no_contiguous_tensor == q.loc).item()

    # it needs copy of tensor
    q = Normal(loc=ones, scale=ones)
    q.load_state_dict(torch.load(save_path))
    assert torch.all(no_contiguous_tensor == q.loc).item()


if __name__ == "__main__":
    # TestReplaceVarDistribution().test_get_entropy()
    test_memoization_for_prior()
