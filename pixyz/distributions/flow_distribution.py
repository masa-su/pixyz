import torch
from ..distributions import Distribution
from ..utils import get_dict_values


class TransformedDistribution(Distribution):
    r"""
    Convert flow transformations to distributions.

    .. math::

        p(z=f_{flow}(x)),

    where :math:`x \sim p_{prior}(x)`.


    Once initializing, it can be handled as a distribution module.

    """

    def __init__(self, prior, flow, var, name="p"):
        if flow.in_features:
            features_shape = [flow.in_features]
        else:
            features_shape = torch.Size()

        super().__init__(var=var, cond_var=prior.cond_var, name=name, features_shape=features_shape)
        self.prior = prior
        self.flow = flow  # FlowList

        self._flow_input_var = prior.var

    @property
    def distribution_name(self):
        return "TransformedDistribution"

    @property
    def flow_input_var(self):
        """list: Input variables of the flow module."""
        return self._flow_input_var

    @property
    def prob_factorized_text(self):
        flow_text = "{}=f_{{flow}}({})".format(self.var[0], self.flow_input_var[0])
        prob_text = "{}({})".format(self._name, flow_text)

        return prob_text

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`forward` or :attr:`update_jacobian` methods to calculate and
        store log-determinant Jacobian.

        """
        return self.flow.logdet_jacobian

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
               compute_jacobian=True):
        # sample from the prior
        sample_dict = self.prior.sample(x_dict, batch_n=batch_n, sample_shape=sample_shape, return_all=return_all)

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

    def forward(self, x, y=None, compute_jacobian=True):
        """
        Forward propagation of flow layers.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor, defaults to None
            Data for conditioning.
        compute_jacobian : bool, defaults to True
            Whether to calculate and store log-determinant Jacobian.
            If true, calculated Jacobian values are stored in :attr:`logdet_jacobian`.

        Returns
        -------
        z : torch.Tensor

        """
        return self.flow.forward(x=x, y=y, compute_jacobian=compute_jacobian)

    def inverse(self, z, y=None):
        """
        Backward (inverse) propagation of flow layers.
        In this method, log-determinant Jacobian is not calculated.

        Parameters
        ----------
        z : torch.Tensor
            Input data.
        y : torch.Tensor, defaults to None
            Data for conditioning.

        Returns
        -------
        x : torch.Tensor

        """
        return self.flow.inverse(z=z, y=y)


class InverseTransformedDistribution(Distribution):
    r"""
    Convert inverse flow transformations to distributions.

    .. math::

        p(x=f^{-1}_{flow}(z)),

    where :math:`z \sim p_{prior}(z)`.

    Once initializing, it can be handled as a distribution module.

    Moreover, this distribution can take a conditional variable.

    .. math::

        p(x=f^{-1}_{flow}(z, y)),

    where :math:`z \sim p_{prior}(z)` and :math:`y` is given.

    """

    def __init__(self, prior, flow, var, cond_var=[], name="p"):
        if flow.in_features:
            features_shape = [flow.in_features]
        else:
            features_shape = torch.Size()

        super().__init__(var, cond_var=cond_var, name=name, features_shape=features_shape)
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
    def prob_factorized_text(self):
        var_text = ','.join(self.flow_output_var + self.cond_var)
        flow_text = "{}=f^{{-1}}_{{flow}}({})".format(self.var[0], var_text)
        prob_text = "{}({})".format(self._name, flow_text)

        return prob_text

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`forward` or :attr:`update_jacobian` methods to calculate and
        store log-determinant Jacobian.

        """
        return self.flow.logdet_jacobian

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False):
        # sample from the prior
        sample_dict = self.prior.sample(x_dict, batch_n=batch_n, sample_shape=sample_shape, return_all=return_all)

        # inverse flow transformation
        _z = get_dict_values(sample_dict, self.flow_output_var)
        _y = get_dict_values(sample_dict, self.cond_var)

        if len(_y) == 0:
            x = self.inverse(_z[0])
        else:
            x = self.inverse(_z[0], y=_y[0])

        output_dict = {self.var[0]: x}

        if return_all:
            sample_dict.update(output_dict)
            return sample_dict

        return output_dict

    def inference(self, x_dict, return_all=True, compute_jacobian=False):
        # flow transformation
        _x = get_dict_values(x_dict, self.var)
        _y = get_dict_values(x_dict, self.cond_var)

        if len(_y) == 0:
            z = self.forward(_x[0], compute_jacobian=compute_jacobian)
        else:
            z = self.forward(_x[0], y=_y[0], compute_jacobian=compute_jacobian)

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

    def forward(self, x, y=None, compute_jacobian=True):
        """
        Forward propagation of flow layers.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor, defaults to None
            Data for conditioning.
        compute_jacobian : bool, defaults to True
            Whether to calculate and store log-determinant Jacobian.
            If true, calculated Jacobian values are stored in :attr:`logdet_jacobian`.

        Returns
        -------
        z : torch.Tensor

        """
        return self.flow.forward(x=x, y=y, compute_jacobian=compute_jacobian)

    def inverse(self, z, y=None):
        """
        Backward (inverse) propagation of flow layers.
        In this method, log-determinant Jacobian is not calculated.

        Parameters
        ----------
        z : torch.Tensor
            Input data.
        y : torch.Tensor, defaults to None
            Data for conditioning.

        Returns
        -------
        x : torch.Tensor

        """
        return self.flow.inverse(z=z, y=y)
