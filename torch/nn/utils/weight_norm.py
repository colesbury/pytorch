"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import functools
import torch


def _decorate(forward, module, norm, name, name_g, name_v):
    @functools.wraps(forward)
    def decorated_forward(*args, **kwargs):
        g = getattr(module, name_g)
        v = getattr(module, name_v)
        w = v * (g / norm(v))
        setattr(module, name, w)
        return forward(*args, **kwargs)
    decorated_forward._norm = norm
    decorated_forward._previous_forward = forward
    return decorated_forward


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

    param = getattr(module, name)

    # construct g,v such that w = g/||v|| * v
    def norm(p):
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

    g = torch.nn.Parameter(norm(param).data)
    v = torch.nn.Parameter(param.data)
    name_g = name + '_g'
    name_v = name + '_v'

    # remove w from parameter list
    del module._parameters[name]

    # add g and v as new parameters and express w as g/||v|| * v
    module.register_parameter(name_g, g)
    module.register_parameter(name_v, v)
    setattr(module, name, v * (g / norm(v)))

    # construct w every time before forward is called
    module.forward = _decorate(module.forward, module, norm, name, name_g, name_v)
    return module


def remove_weight_norm(module, name='weight'):
    """Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of the weight-normalized parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> m.weight_g
        >>> remove_weight_norm(m)
    """

    name_g = name + '_g'
    name_v = name + '_v'
    g = getattr(module, name_g)
    v = getattr(module, name_v)

    norm = module.forward._norm
    w = v * (g / norm(v))

    del module._parameters[name_g]
    del module._parameters[name_v]
    delattr(module, name)
    module.register_parameter(name, torch.nn.Parameter(w.data))

    module.forward = module.forward._previous_forward
    return module
