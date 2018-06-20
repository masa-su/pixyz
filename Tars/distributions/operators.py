from torch import nn

from ..utils import get_dict_values


class MultiplyDistributionModel(nn.Module):
    """
    p(x,y|z) = p(x|z,y)p(y|z)

    Paramaters
    -------
    A : DistributionModel or MultiplyDistributionModel
    B : DistributionModel or MultiplyDistributionModel

    Examples
    --------
    >>> p_multi = MultipleDistributionModel([A, B])
    >>> p_multi = A * B

    TODO: how about add_module?
    """

    def __init__(self, A, B):
        super(MultiplyDistributionModel, self).__init__()
        """
        Set parents and children
        If "inherited variables" are exist (which means p(e|c)p(c|a,b)),
        then A is a child and B is a parent.
        Else (which means p(c|a,b)p(e|c) or p(c|a)p(b|a)),
        then A is a parent and B is a child.
        """
        _vars = A.cond_var + B.var
        _inh_var = [item for item in set(_vars) if _vars.count(item) > 1]
        if len(_inh_var) > 0:
            self.parents_var = B
            self.children_var = A
        else:
            self.parents_var = A
            self.children_var = B

        # set variables
        _var = self.children_var.var + self.parents_var.var
        var = sorted(set(_var), key=_var.index)

        # set conditional variables
        _cond_var = self.children_var.cond_var + self.parents_var.cond_var
        cond_var = sorted(set(_cond_var), key=_cond_var.index)

        # check the conflict in conditional variables
        all_var = var + cond_var
        inh_var = [item for item in set(all_var) if all_var.count(item) > 1]
        cond_var = [item for item in cond_var if item not in inh_var]

        self.var = var
        self.cond_var = cond_var
        self.inh_var = inh_var

        self.prob_factorized_text = self.children_var.prob_factorized_text + \
            self.parents_var.prob_text

        if len(self.cond_var) == 0:
            self.prob_text = "p(" + ','.join(self.var) + ")"
        else:
            self.prob_text = "p(" + ','.join(self.var) + \
                "|" + ','.join(self.cond_var) + ")"

    def sample(self, x=None, batch_size=1, return_all=True, *args, **kwargs):
        # input : dict
        # output : dict

        # sample from the parent distribution
        if x is None:
            if len(self.parents_var.cond_var) > 0:
                raise ValueError("You should set inputs.")

            parents_output = self.parents_var.sample(batch_size=batch_size)

        else:
            if batch_size == 1:
                batch_size = list(x.values())[0].shape[0]

            if list(x.values())[0].shape[0] != batch_size:
                raise ValueError("Invalid batch size")

            if set(list(x.keys())) != set(self.cond_var):
                raise ValueError("Input's keys are not valid.")

            if len(self.parents_var.cond_var) > 0:
                parents_input = get_dict_values(
                    x, self.parents_var.cond_var, return_dict=True)
                parents_output = self.parents_var.sample(
                    parents_input, return_all=False)
            else:
                parents_output = self.parents_var.sample(
                    batch_size=batch_size, return_all=False)

        # sample from the child distribution
        children_input_inh = get_dict_values(
            parents_output, self.inh_var, return_dict=True)
        if x is None:
            children_input = children_input_inh
        else:
            children_cond_exc_inh = list(
                set(self.children_var.cond_var)-set(self.inh_var))
            children_input = get_dict_values(
                x, children_cond_exc_inh, return_dict=True)
            children_input.update(children_input_inh)

        children_output = self.children_var.sample(
            children_input, return_all=False)

        output = parents_output
        output.update(children_output)

        if return_all and x:
            output.update(x)

        return output

    def sample_mean(self, x=None, batch_size=1, *args, **kwargs):
        # input : dict
        # output : dict

        # sample from the parent distribution
        if x is None:
            if len(self.parents_var.cond_var) > 0:
                raise ValueError("You should set inputs.")

            parents_output = self.parents_var.sample(batch_size=batch_size)

        else:
            if batch_size == 1:
                batch_size = list(x.values())[0].shape[0]

            if list(x.values())[0].shape[0] != batch_size:
                raise ValueError("Invalid batch size")

            if set(list(x.keys())) != set(self.cond_var):
                raise ValueError("Input's keys are not valid.")

            if len(self.parents_var.cond_var) > 0:
                parents_input = get_dict_values(
                    x, self.parents_var.cond_var, return_dict=True)
                parents_output = self.parents_var.sample(
                    parents_input, return_all=False)
            else:
                parents_output = self.parents_var.sample(
                    batch_size=batch_size, return_all=False)

        # sample from the child distribution
        children_input_inh = get_dict_values(
            parents_output, self.inh_var, return_dict=True)
        if x is None:
            children_input = children_input_inh
        else:
            children_cond_exc_inh = list(
                set(self.children_var.cond_var)-set(self.inh_var))
            children_input = get_dict_values(
                x, children_cond_exc_inh, return_dict=True)
            children_input.update(children_input_inh)

        output = self.children_var.sample_mean(children_input)
        return output

    def log_likelihood(self, x):
        # input : dict
        # output : dict

        parents_x = get_dict_values(
            x, self.parents_var.cond_var + self.parents_var.var,
            return_dict=True)
        children_x = get_dict_values(
            x, self.children_var.cond_var + self.children_var.var,
            return_dict=True)

        return self.parents_var.log_likelihood(parents_x) +\
            self.children_var.log_likelihood(children_x)

    def forward(self, *args, **kwargs):
        NotImplementedError

    def __mul__(self, other):
        return MultiplyDistributionModel(self, other)
