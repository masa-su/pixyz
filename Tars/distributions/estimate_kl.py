import torch

from ..utils import get_dict_values

def analytical_kl(q1, q2, given):
    # TODO: change to a class and add a method for estimate kl
    try:
        [x1, x2] = given
    except:
        raise ValueError("The length of given list must be 2, "
                         "got %d" % len(given))

    if (q1.distribution_name == "Gaussian") and (q2.distribution_name == "UnitGaussian"):
        inputs = get_dict_values(x1, q1.cond_var)
        mu, sigma = q1.forward(*inputs)
        return gauss_unitgauss_kl(mu, sigma)

    raise Exception("You cannot use this distribution as q or prior, "
                    "got %s and %s." % (q1.distribution_name, q2.distribution_name))
        

def gauss_unitgauss_kl(mu, sigma, unit_mu=0, unit_sigma=1):
    # TODO: fix this output to one that considers unit_mu and unit_sigma
    return -0.5 * torch.sum(1 + torch.log(sigma) - mu**2 - sigma, dim=1)
