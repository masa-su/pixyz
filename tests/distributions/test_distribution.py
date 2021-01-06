import torch
from pixyz.distributions import Normal


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

        # dist = Normal(var=['x'], cond_var=['y'], loc='y', scale=1) * FactorizedBernoulli(
        #     var=['y'], probs=torch.tensor([0.3, 0.8]))
        # dist.graph.set_option(dict(batch_n=3, sample_shape=(4,)), ['y'])
        # sample = dist.sample()
        # assert sample['y'].shape == torch.Size([4, 3, 2])
        # assert sample['x'].shape == torch.Size([4, 3, 2])
        # dist.graph.set_option(dict(), ['y'])
        # dist.graph.set_option(dict(sum_features=True, feature_dims=[-1]))
        # assert dist.get_log_prob(sample).shape == torch.Size([4, 3])


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
