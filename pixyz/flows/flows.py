from torch import nn


class Flow(nn.Module):
    """Flow class. In Pixyz, all flows are required to inherit this class."""

    def __init__(self, in_features):
        """
        Parameters
        ----------
        in_features : int
            Size of  input data.

        """
        super().__init__()
        self._in_features = in_features
        self._logdet_jacobian = None

    @property
    def in_features(self):
        return self._in_features

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
        z = x
        return z

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
        x = z
        return x

    @property
    def logdet_jacobian(self):
        """
        Get log-determinant Jacobian.

        Before calling this, you should run :attr:`forward` or :attr:`update_jacobian` methods to calculate and
        store log-determinant Jacobian.

        """
        return self._logdet_jacobian


class FlowList(Flow):

    def __init__(self, flow_list):
        """
        Hold flow modules in a list.

        Once initializing, it can be handled as a single flow module.

        Notes
        -----
        Indexing is not supported for now.


        Parameters
        ----------
        flow_list : list

        """
        super().__init__(flow_list[0].in_features)
        self.flow_list = nn.ModuleList(flow_list)

    def forward(self, x, y=None, compute_jacobian=True):
        logdet_jacobian = 0

        for flow in self.flow_list:
            x = flow.forward(x, y, compute_jacobian)
            if compute_jacobian:
                logdet_jacobian = logdet_jacobian + flow.logdet_jacobian

        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian

        return x

    def inverse(self, z, y=None):
        for flow in self.flow_list[::-1]:
            z = flow.inverse(z, y)
        return z

    def __repr__(self):
        # rename "ModuleList" to "FlowList"
        flow_list_repr = self.flow_list.__repr__().replace("ModuleList", "FlowList")
        return flow_list_repr
