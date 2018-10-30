from abc import ABCMeta
import torch


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, distributions):
        self.loss_cls = None
        self.test_loss_cls = None
        self.optimizer = None

        self.distributions = distributions

    def __str__(self):
        prob_text = [prob.prob_text for prob in self.distributions._modules.values()]

        text = "Distributions (for training): \n  {} \n".format(", ".join(prob_text))
        text += "Loss function: \n  {}".format(str(self.loss_cls))
        return text

    def train(self, train_x, adversarial_loss=False, **kwargs):
        self.distributions.train()

        self.optimizer.zero_grad()
        loss = self.loss_cls.estimate(train_x, **kwargs)

        # backprop
        loss.backward()

        # update params
        self.optimizer.step()

        # train the adversarial loss function (only for adversarial training)
        if adversarial_loss:
            d_loss = self.loss_cls.train(train_x, **kwargs)
            return loss, d_loss

        return loss

    def test(self, test_x, adversarial_loss=False, **kwargs):
        self.distributions.eval()

        with torch.no_grad():
            loss = self.test_loss_cls.estimate(test_x, **kwargs)
            if adversarial_loss:
                d_loss = self.loss_cls.test(test_x, **kwargs)
                return loss, d_loss

        return loss
