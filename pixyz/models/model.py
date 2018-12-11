from torch import optim, nn
import torch
import re

from ..utils import tolist


class Model(object):
    def __init__(self, loss,
                 test_loss=None,
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={}):

        # set losses
        self.loss_cls = None
        self.test_loss_cls = None
        self.set_loss(loss, test_loss)

        # set distributions (for training)
        self.distributions = nn.ModuleList(tolist(distributions))

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

    def __str__(self):
        prob_text = [prob.prob_text for prob in self.distributions._modules.values()]

        text = "Distributions (for training): \n  {} \n".format(", ".join(prob_text))
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

    def train(self, train_x={}, **kwargs):
        self.distributions.train()

        self.optimizer.zero_grad()
        loss = self.loss_cls.estimate(train_x, **kwargs)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return loss

    def test(self, test_x={}, **kwargs):
        self.distributions.eval()

        with torch.no_grad():
            loss = self.test_loss_cls.estimate(test_x, **kwargs)

        return loss
