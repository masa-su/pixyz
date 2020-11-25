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
