from torch import optim, nn
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import re

from ..utils import tolist
from ..distributions.distributions import Distribution


class Model(object):
    def __init__(self, loss,
                 test_loss=None,
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None, clip_grad_value=None):

        # set losses
        self.loss_cls = None
        self.test_loss_cls = None
        self.set_loss(loss, test_loss)

        # set distributions (for training)
        self.distributions = nn.ModuleList(tolist(distributions))

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

        self.clip_norm = clip_grad_norm
        self.clip_value = clip_grad_value

    def __str__(self):
        prob_text = []
        func_text = []

        for prob in self.distributions._modules.values():
            if isinstance(prob, Distribution):
                prob_text.append(prob.prob_text)
            else:
                func_text.append(prob.__str__())

        text = "Distributions (for training): \n  {} \n".format(", ".join(prob_text))
        if len(func_text) > 0:
            text += "Deterministic functions (for training): \n  {} \n".format(", ".join(func_text))

        text += "Loss function: \n  {} \n".format(str(self.loss_cls))
        optimizer_text = re.sub('^', ' ' * 2, str(self.optimizer), flags=re.MULTILINE)
        text += "Optimizer: \n{}".format(optimizer_text)
        return text

    def set_loss(self, loss, test_loss=None):
        self.loss_cls = loss
        if test_loss:
            self.test_loss_cls = test_loss
        else:
            self.test_loss_cls = loss

    def train(self, train_x_dict={}, **kwargs):
        self.distributions.train()

        self.optimizer.zero_grad()
        loss = self.loss_cls.eval(train_x_dict, **kwargs)

        # backprop
        loss.backward()

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.optimizer.step()

        return loss

    def test(self, test_x_dict={}, **kwargs):
        self.distributions.eval()

        with torch.no_grad():
            loss = self.test_loss_cls.eval(test_x_dict, **kwargs)

        return loss
