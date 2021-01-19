import torch
from pixyz.losses import IterativeLoss, Parameter, Expectation
from pixyz.distributions import Normal


class TestIterativeLoss:
    def test_print_latex(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.loss_text == r"\sum_{t=0}^{" + str(t_max - 1) + "} t"

    def test_time_specific_step_loss(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.eval() == sum(range(t_max))

    def test_input_var(self):
        q = Normal(var=['z'], cond_var=['x'], loc='x', scale=1)
        p = Normal(var=['y'], cond_var=['z'], loc='z', scale=1)
        e = Expectation(q, p.log_prob())
        assert set(e.input_var) == set(('x', 'y'))
        assert e.eval({'y': torch.zeros(1), 'x': torch.zeros(1)}).shape == torch.Size([1])
