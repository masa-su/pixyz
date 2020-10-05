import pytest
from os.path import join as pjoin
import torch
from pixyz.utils import lru_cache_for_sample_dict
from pixyz.distributions.exponential_distributions import RelaxedBernoulli, Normal
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE


@pytest.fixture(scope="module", autouse=True)
def train_loader():
    # prepare object
    yield None
    # terminate object


@pytest.mark.parametrize(
    "x, y", [
        ("aaa", "bbb"),
        ("aaa", "aaa"),
        ("bbb", "bbb")
    ]
)
@pytest.mark.small
def test_something(train_loader, x, y, tmpdir):
    print(tmpdir)
    assert True


def test_mytest():
    with pytest.raises(RuntimeError) as excinfo:
        def f():
            f()

        f()
    assert "maximum recursion" in str(excinfo.value)


@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record


def test_customer_records(make_customer_record):
    # customer_1 = make_customer_record("Lisa")
    # customer_2 = make_customer_record("Mike")
    # customer_3 = make_customer_record("Meredith")
    pass


def test_memoization():
    exec_order = []

    class Encoder(Normal):
        def __init__(self, exec_order):
            super().__init__(cond_var=["x"], var=["z"], name="q")
            self.linear = torch.nn.Linear(10, 10)
            self.exec_order = exec_order

        @lru_cache_for_sample_dict(maxsize=2)
        def get_params(self, params_dict={}, **kwargs):
            return super().get_params(params_dict, **kwargs)

        def forward(self, x):
            exec_order.append("E")
            return {"loc": self.linear(x), "scale": 1.0}

    class Decoder(Normal):
        def __init__(self, exec_order):
            super().__init__(cond_var=["z"], var=["x"], name="p")
            self.exec_order = exec_order

        @lru_cache_for_sample_dict(maxsize=2)
        def get_params(self, params_dict={}, **kwargs):
            return super().get_params(params_dict, **kwargs)

        def forward(self, z):
            self.exec_order.append("D")
            return {"loc": z, "scale": 1.0}

    def prior():
        return Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[10], name="p_{prior}")

    q = Encoder(exec_order)
    p = Decoder(exec_order)

    prior = prior()
    kl = KullbackLeibler(q, prior)

    mdl = VAE(q, p, regularizer=kl, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3})

    x = torch.zeros((10, 10))
    mdl.train({"x": x})
    assert exec_order == ["E", "D"]


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


class TestNormal:
    def test_init_with_same_param(self):
        n = Normal(var=['x'], cond_var=['y'], loc='y', scale='y')
        result = n.sample({'y': torch.ones(2, 3)})
        assert result['x'].shape == (2, 3)


class TestRelaxedBernoulli:
    def test_log_prob_of_hard_value(self):
        rb = RelaxedBernoulli(var=['x'], probs=torch.ones(2), temperature=torch.tensor(0.5))
        assert self.nearly_eq(rb.get_log_prob({'x': torch.tensor([0., 1.])}), torch.tensor([-15.9424]))

    def nearly_eq(self, tensor1, tensor2):
        return abs(tensor1.item() - tensor2.item()) < 0.001


class TestIterativeLoss:
    def test_restore_given_value_after_eval(self):
        pass


if __name__ == "__main__":
    test_save_dist(".", torch.zeros(2, 3))
