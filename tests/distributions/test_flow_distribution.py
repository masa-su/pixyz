import torch
from pixyz.distributions import Normal, TransformedDistribution
from pixyz.flows import FlowList, PlanarFlow


class TestTransformedDistribution:
    def test_sample_bypass_option(self):
        x_dim = 2
        prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["x"], features_shape=[x_dim], name="prior")
        f = FlowList([PlanarFlow(x_dim) for _ in range(32)])
        q = TransformedDistribution(prior, f, var=["z"], name="q")
        sample = q.sample(bypass_from="loc")
        assert torch.equal(sample['x'], torch.tensor([[0., 0.]]))
        assert sample['z'].shape == torch.Size([1, x_dim])
