import torch
from torch import nn
from .distributions import Distribution


class QuantizedDistribution(Distribution):
    def __init__(self, var, cond_var, k, d, name="p", features_shape=torch.Size()):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape)
        self.embedding = nn.Embedding(k, d)
        self.embedding.weight.data.uniform_(-1./k, 1./k)

    def forward(self, *args, **kwargs):
        pass


