from ..distributions import Distribution
from ..utils import get_dict_values


class FlowDistribution(Distribution):
    """
    p(z)
    z = f(x), x~p(x)
    """

    def __init__(self, prior, flows, var, name="p"):

        super().__init__(var, cond_var=prior.cond_var, name=name, dim=flows[-1].in_features)
        self.prior = prior
        self.flows = flows  # FlowList

        self._flow_input_var = prior.var

    @property
    def flow_input_var(self):
        return self._flow_input_var

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`sample` methods.

        """
        return self._logdet_jacobian

    def sample(self, x_dict={}, return_all=True, compute_jacobian=True):
        # sample from the prior
        sample_dict = self.prior.sample(x_dict, return_all=True)

        # flow transformation
        _x = get_dict_values(sample_dict, self.input_var)
        z = self.forward(_x, compute_jacobian=compute_jacobian)
        output_dict = {self.var[0]: z}

        if return_all:
            sample_dict.update(output_dict)
            return sample_dict

        return output_dict

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        # prior
        log_prob = self.prior.get_log_prob(x_dict, sum_features=sum_features, feature_dims=feature_dims)

        # flow
        log_prob -= self.logdet_jacobian

        return log_prob

    def forward(self, *args, **kwargs):
        return self.flows.forward(*args, **kwargs)


class InverseFlowDistribution(Distribution):
    """
    p(x)
    x = f^-1(z), z~p(z)
    """

    def __init__(self, prior, flows, var, name="p"):

        super().__init__(var, cond_var=[], name=name, dim=flows[0].in_features)
        self.prior = prior
        self.flows = flows  # FlowList

        self._flow_output_var = prior.var

    @property
    def flow_output_var(self):
        return self._flow_output_var

    def sample(self, z_dict={}, return_all=True, **kwargs):
        # sample from the prior
        sample_dict = self.prior.sample(z_dict, return_all=True)

        # inverse flow transformation
        _z = get_dict_values(sample_dict, self.flow_output_var)
        x = self.forward(_z, inverse=True)
        output_dict = {self.var[0]: x}

        if return_all:
            sample_dict.update(output_dict)
            return sample_dict

        return output_dict

    def inference(self, x_dict, return_all=True, compute_jacobian=False):
        # flow transformation
        _x = get_dict_values(x_dict, self.input_var)
        z = self.forward(_x, compute_jacobian=compute_jacobian)
        output_dict = {self.flow_output_var[0]: z}

        if return_all:
            output_dict.update(x_dict)

        return output_dict

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None):
        # flow
        output_dict = self.inference(x_dict, return_all=True, compute_jacobian=True)
        log_prob = self.logdet_jacobian

        # prior
        log_prob += self.prior.get_log_prob(output_dict, sum_features=sum_features, feature_dims=feature_dims)

        return log_prob

    def forward(self, *args, **kwargs):
        return self.flows.forward(*args, **kwargs)
