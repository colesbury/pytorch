"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter


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
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return v * (g / norm(v, self.dim))

    @staticmethod
    def apply(module, name, dim):
        fn = WeightNorm(name, dim)

        w = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm(w, dim).data))
        module.register_parameter(name + '_v', Parameter(w.data))
        setattr(module, name, fn.compute_weight(module))

        handle = hooks.RemovableHandle(module._forward_pre_hooks)
        module._forward_pre_hooks[handle.id] = fn

        return fn

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


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
    WeightNorm.apply(module, name, dim)
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
