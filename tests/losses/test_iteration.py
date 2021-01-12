
from pixyz.losses import IterativeLoss, Parameter


class TestIterativeLoss:
    def test_print_latex(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.loss_text == r"\sum_{t=0}^{" + str(t_max - 1) + "} t"

    def test_time_specific_step_loss(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.eval() == sum(range(t_max))
