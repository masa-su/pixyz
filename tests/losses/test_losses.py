from pixyz.distributions import Normal
from pixyz.losses import Expectation


class TestExpectation:
    def test_sample_mean(self):
        p = Normal(loc=0, scale=1)
        f = p.log_prob()
        e = Expectation(p, f)
        e.eval({}, sample_mean=True)
