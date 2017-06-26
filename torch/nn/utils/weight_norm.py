"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import torch


def norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    if dim != 0:
        p = p.transpose(0, dim)
    output_size = (p.size(0),) + (1,) * (p.dim() - 1)
    p = p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    if dim != 0:
        p = p.transpose(0, dim)
    return p


class WeightNorm(object):
    def __init__(self, module, name, dim):
        self.module = module
        self.name = name
        self.dim = dim
        self.name_g = name + '_g'
        self.name_v = name + '_v'

        w = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(self.name_g, torch.nn.Parameter(norm(w, dim).data))
        module.register_parameter(self.name_v, torch.nn.Parameter(w.data))
        setattr(module, name, self.compute_weight())

        self._forward = self.module.forward

    def compute_weight(self):
        g = getattr(self.module, self.name_g)
        v = getattr(self.module, self.name_v)
        return v * (g / norm(v, self.dim))

    def wrapped_forward(self, *args, **kwargs):
        setattr(self.module, self.name, self.compute_weight())
        return self._forward(*args, **kwargs)

    def remove(self):
        w = self.compute_weight()
        del self.module._parameters[self.name_g]
        del self.module._parameters[self.name_v]
        delattr(self.module, self.name)
        self.module.register_parameter(self.name, torch.nn.Parameter(w.data))
        self.module.forward = self._forward


def weight_norm(module, name='weight', dim=0):
    """Applies weight normalization to a parameter in the given module.

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    The module's forward function is wrapped to first recompute the weight
    tensor from the magnitude and direction.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of the parameter to reparameterize
        dim (int, optional): todo

    Example::

        >>> m = weight_norm(nn.Linear(20, 40))
        >>> m.weight_g.size()
        torch.Size([1])
        >>> m.weight_v.size()
        torch.Size([40, 20])
    """
    module.forward = WeightNorm(module, name, dim).wrapped_forward
    return module


def remove_weight_norm(module):
    """Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """

    wn = module.forward.__self__
    if not isinstance(wn, WeightNorm):
        raise TypeError('expected WeightNorm (got {})'.format(type(wn).__name__))

    wn.remove()
    return module
