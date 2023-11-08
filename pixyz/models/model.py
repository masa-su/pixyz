from torch import optim, nn
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import re

from ..utils import tolist
from ..distributions.distributions import Distribution


class Model(object):
    """
    This class is for training and testing a loss class.
    It requires a defined loss class, distributions to train, and optimizer for initialization.

    Examples
    --------
    >>> import torch
    >>> from torch import optim
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Bernoulli, Normal
    >>> from pixyz.losses import KullbackLeibler
    ...
    >>> # Set distributions (Distribution API)
    >>> class Inference(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x"],name="q")
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    ...
    >>> class Generator(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(var=["x"],cond_var=["z"],name="p")
    ...         self.model = torch.nn.Linear(64, 128)
    ...     def forward(self, z):
    ...         return {"probs": torch.sigmoid(self.model(z))}
    ...
    >>> p = Generator()
    >>> q = Inference()
    >>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
    ...                var=["z"], features_shape=[64], name="p_{prior}")
    ...
    >>> # Define a loss function (Loss API)
    >>> reconst = -p.log_prob().expectation(q)
    >>> kl = KullbackLeibler(q,prior)
    >>> loss_cls = (reconst - kl).mean()
    >>> print(loss_cls)
    mean \\left(- D_{KL} \\left[q(z|x)||p_{prior}(z) \\right] - \\mathbb{E}_{q(z|x)} \\left[\\log p(x|z) \\right] \\right)
    >>>
    >>> # Set a model (Model API)
    >>> model = Model(loss=loss_cls, distributions=[p, q],
    ...               optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    >>> # Train and test the model
    >>> data = torch.randn(1, 128)  # Pseudo data
    >>> train_loss = model.train({"x": data})
    >>> test_loss = model.test({"x": data})

    """

    def __init__(self, loss,
                 test_loss=None,
                 distributions=[],
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None,
                 retain_graph=False,
                 use_amp=False):
        """
        Parameters
        ----------
        loss : pixyz.losses.Loss
            Loss class for training.
        test_loss : pixyz.losses.Loss
            Loss class for testing.
        distributions : list
            List of :class:`pixyz.distributions.Distribution`.
        optimizer : torch.optim
            Optimization algorithm.
        optimizer_params : dict
            Parameters of optimizer
        clip_grad_norm : float or int
            Maximum allowed norm of the gradients.
        clip_grad_value : float or int
            Maximum allowed value of the gradients.
        retain_graph : bool
            If False, the graph used to compute the grads will be freed.
        use_amp : bool
            If True, use automatic mixed precision (AMP).
        """

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
        self.retain_graph = retain_graph
        self.use_amp = use_amp

        # set scaler for amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def __str__(self):
        prob_text = []
        func_text = []

        for prob in self.distributions._modules.values():
            if isinstance(prob, Distribution):
                prob_text.append(prob.prob_text)
            else:
                func_text.append(prob.__str__())

        text = "Distributions (for training):\n  {}\n".format(", ".join(prob_text))
        if len(func_text) > 0:
            text += "Deterministic functions (for training):\n  {}\n".format(", ".join(func_text))

        text += "Loss function:\n  {}\n".format(str(self.loss_cls))
        optimizer_text = re.sub('^', ' ' * 2, str(self.optimizer), flags=re.MULTILINE)
        text += "Optimizer:\n{}".format(optimizer_text)
        return text

    def set_loss(self, loss, test_loss=None):
        self.loss_cls = loss
        if test_loss:
            self.test_loss_cls = test_loss
        else:
            self.test_loss_cls = loss

    def train(self, train_x_dict={}, **kwargs):
        """Train the model.

        Parameters
        ----------
        train_x_dict : dict
            Input data.
        **kwargs

        Returns
        -------
        loss : torch.Tensor
            Train loss value

        """
        self.distributions.train()

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss = self.loss_cls.eval(train_x_dict, **kwargs)

        # backprop
        self.scaler.scale(loss).backward(retain_graph=self.retain_graph)

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.scaler.step(self.optimizer)

        self.scaler.update()

        return loss

    def test(self, test_x_dict={}, **kwargs):
        """Test the model.

        Parameters
        ----------
        test_x_dict : dict
            Input data
        **kwargs

        Returns
        -------
        loss : torch.Tensor
            Test loss value

        """
        self.distributions.eval()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            loss = self.test_loss_cls.eval(test_x_dict, **kwargs)

        return loss

    def save(self, path):
        """Save the model. The only parameters that are saved are those that are included in the distribution.
         Parameters such as device, optimizer, placement of clip_grad, etc. are not saved.

        Parameters
        ----------
        path : str
            Target file path

        """
        torch.save({
            'distributions': self.distributions.state_dict(),
        }, path)

    def load(self, path):
        """Load the model.

        Parameters
        ----------
        path : str
            Target file path

        """
        checkpoint = torch.load(path)
        self.distributions.load_state_dict(checkpoint['distributions'])
