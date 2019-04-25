from ..distributions import Distribution
from ..utils import get_dict_values

import math
class TransformedDistribution(Distribution):
    """
    p(z)
    z = f(x), x~p(x)
    """

    def __init__(self, prior, flow, var, name="p"):

        super().__init__(var=var, cond_var=prior.cond_var, name=name, dim=flow.in_features)
        self.prior = prior
        self.flow = flow  # FlowList

        self._flow_input_var = prior.var

    @property
    def distribution_name(self):
        return "TransformedDistribution"

    @property
    def flow_input_var(self):
        return self._flow_input_var

    @property
    def logdet_jacobian(self):
        """Get log-determinant Jacobian."""
        return self.flow.logdet_jacobian

    def sample(self, x_dict={}, shape=None, batch_size=1, return_all=True, compute_jacobian=True, **kwargs):
        # sample from the prior
        sample_dict = self.prior.sample(x_dict, shape=shape, batch_size=batch_size, return_all=True, **kwargs)

        # flow transformation
        _x = get_dict_values(sample_dict, self.flow_input_var)[0]
        z = self.forward(_x, compute_jacobian=compute_jacobian)
        output_dict = {self.var[0]: z}

        if return_all:
            sample_dict.update(output_dict)
            return sample_dict

        return output_dict

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, compute_jacobian=False):
        # prior
        log_prob_prior = self.prior.get_log_prob(x_dict, sum_features=sum_features, feature_dims=feature_dims)

        # flow
        if compute_jacobian:
            self.sample(x_dict, return_all=False, compute_jacobian=True)

        return log_prob_prior - self.logdet_jacobian

    def forward(self, *args, **kwargs):
        return self.flow.forward(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        return self.flow.inverse(*args, **kwargs)


class InverseTransformedDistribution(Distribution):
    """
    p(x)
    x = f^-1(z), z~p(z)
    """

    def __init__(self, prior, flow, var, name="p"):

        super().__init__(var, cond_var=[], name=name, dim=flow.in_features)
        self.prior = prior
        self.flow = flow  # FlowList

        self._flow_output_var = prior.var

    @property
    def distribution_name(self):
        return "InverseTransformedDistribution"

    @property
    def flow_output_var(self):
        return self._flow_output_var

    @property
    def logdet_jacobian(self):
        """Get log-determinant Jacobian."""
        return self.flow.logdet_jacobian

    def sample(self, z_dict={}, shape=None, batch_size=1, return_all=True, **kwargs):
        # sample from the prior
        sample_dict = self.prior.sample(z_dict, shape=shape, batch_size=batch_size, return_all=True, **kwargs)

        # inverse flow transformation
        _z = get_dict_values(sample_dict, self.flow_output_var)[0]
        x = self.inverse(_z)
        output_dict = {self.var[0]: x}

        if return_all:
            sample_dict.update(output_dict)
            return sample_dict

        return output_dict

    def inference(self, x_dict, return_all=True, compute_jacobian=False):
        # flow transformation
        _x = get_dict_values(x_dict, self.var)[0]
        z = self.forward(_x, compute_jacobian=compute_jacobian)
        output_dict = {self.flow_output_var[0]: z}

        if return_all:
            output_dict.update(x_dict)

        return output_dict

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        # flow
        output_dict = self.inference(x_dict, return_all=True, compute_jacobian=True)

        # prior
        log_prob_prior = self.prior.get_log_prob(output_dict, sum_features=sum_features, feature_dims=feature_dims)

        return log_prob_prior + self.logdet_jacobian

    def forward(self, *args, **kwargs):
        return self.flow.forward(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        return self.flow.inverse(*args, **kwargs)
