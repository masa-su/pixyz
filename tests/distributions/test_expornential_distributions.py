import torch

from pixyz.distributions.exponential_distributions import RelaxedBernoulli, Normal


class TestNormal:
    def test_init_with_same_param(self):
        n = Normal(var=['x'], cond_var=['y'], loc='y', scale='y')
        result = n.sample({'y': torch.ones(2, 3)})
        assert result['x'].shape == (2, 3)


class TestRelaxedBernoulli:
    def test_log_prob_of_hard_value(self):
        rb = RelaxedBernoulli(var=['x'], temperature=torch.tensor(0.5), probs=torch.ones(2))
        assert self.nearly_eq(rb.get_log_prob({'x': torch.tensor([0., 1.])}), torch.tensor([-15.9424]))

    def nearly_eq(self, tensor1, tensor2):
        return abs(tensor1.item() - tensor2.item()) < 0.001
