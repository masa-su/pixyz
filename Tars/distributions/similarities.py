import torch
from torch import nn

from ..utils import get_dict_values


class SimilarityLoss(object):
    """
    Learning Modality-Invariant Representations for Speech and Images (Leidai et. al.)
    """
    def __init__(self, p1, p2, var=["z"], margin=0):
        self.p1 = p1
        self.p2 = p2
        self.var = var
        self.loss = nn.MarginRankingLoss(margin=margin, reduce=False)

    def _sim(self, x1, x2):
        return torch.sum(x1*x2, dim=1)

    def estimate(self, x):
        inputs = get_dict_values(x, self.p1.cond_var, True)
        sample1 = get_dict_values(self.p1.sample(inputs), self.var)[0]

        inputs = get_dict_values(x, self.p2.cond_var, True)
        sample2 = get_dict_values(self.p2.sample(inputs), self.var)[0]

        batch_size = sample1.shape[0]
        shuffle_id = torch.randperm(batch_size)
        _sample1 = sample1[shuffle_id]
        _sample2 = sample2[shuffle_id]

        sim12 = self._sim(sample1, sample2)
        sim1_2 = self._sim(sample1, _sample2)
        sim_12 = self._sim(_sample1, sample2)

        dummy_label = torch.ones_like(sim12)
        loss = self.loss(sim12, sim1_2, dummy_label) \
            + self.loss(sim12, sim_12, dummy_label)

        return loss


class MultiModalContrastivenessLoss(object):
    """
    Disentangling by Partitioning:
    A Representation Learning Framework for Multimodal Sensory Data
    """
    def __init__(self, p1, p2, margin=0.5):
        self.p1 = p1
        self.p2 = p2
        self.loss = nn.MarginRankingLoss(margin=margin)

    def _sim(self, x1, x2):
        return torch.exp(-torch.norm(x1-x2, 2, dim=1) / 2)

    def estimate(self, x):
        inputs = get_dict_values(x, self.p1.cond_var, True)
        sample1 = self.p1.sample_mean(inputs)

        inputs = get_dict_values(x, self.p2.cond_var, True)
        sample2 = self.p2.sample_mean(inputs)

        batch_size = sample1.shape[0]
        shuffle_id = torch.randperm(batch_size)
        _sample1 = sample1[shuffle_id]
        _sample2 = sample2[shuffle_id]

        sim12 = self._sim(sample1, sample2)
        sim1_2 = self._sim(sample1, _sample2)
        sim_12 = self._sim(_sample1, sample2)

        dummy_label = torch.ones_like(sim12)
        loss = self.loss(sim12, sim1_2, dummy_label) \
            + self.loss(sim12, sim_12, dummy_label)

        return loss
