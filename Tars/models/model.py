from abc import ABCMeta
import torch


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.loss_cls = None
        self.test_loss_cls = None
        self.optimizer = None

    def train(self, train_x, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss_cls.estimate(train_x, **kwargs)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        return loss

    def test(self, test_x, **kwargs):
        with torch.no_grad():
            loss = self.test_loss_cls.estimate(test_x, **kwargs)

        return loss
