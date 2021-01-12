import os
import torch
import torch.nn as nn
from pixyz.distributions import Normal
from pixyz.losses import CrossEntropy
from pixyz.models import Model


class TestModel:
    def _make_model(self, loc):
        class Dist(Normal):
            def __init__(self):
                super().__init__(loc=loc, scale=1)
                self.module = nn.Linear(2, 2)

        p = Dist()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        loss = CrossEntropy(p, p).to(device)
        model = Model(loss=loss, distributions=[p])
        return model

    def test_save_load(self, tmp_path):
        model = self._make_model(0)
        save_path = os.path.join(tmp_path, 'model.pth')
        model.save(save_path)

        model = self._make_model(1)
        p: Normal = model.distributions[0]
        assert p.get_params()['loc'] == 1

        model.load(save_path)
        p: Normal = model.distributions[0]
        assert p.get_params()['loc'] == 0
