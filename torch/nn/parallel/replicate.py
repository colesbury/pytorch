from copy import copy
from collections import OrderedDict

from ..modules import Module
import torch.cuda.comm as comm
from torch.autograd import Variable
import torch


def _replicate_module(module, gpu, param_remap):
    if module is None:
        return module
    replica = copy(module)
    replica._parameters = OrderedDict()
    for key, param in module._parameters.items():
        replica._parameters[key] = param_remap.get(param)
    replica._buffers = {}
    for key, buffer in module._buffers.items():
        replica._buffers[key] = param_remap.get(buffer)
    if replica._modules:
        replica._modules = OrderedDict()
        for name, child in module._modules.items():
            replica._modules[name] = _replicate_module(child, gpu, param_remap)
    return replica

def parameters_and_buffers(network):
    params = []
    indices = {}
    memo = set()
    for m in network.modules():
        for p in m._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                indices[p] = len(params)
                params.append(p)
        for p in m._buffers.values():
            if p is not None and p not in memo:
                memo.add(p)
                indices[p] = len(params)
                params.append(Variable(p))
    return params, indices


import wrapt
import weakref

class ModuleProxy(wrapt.CallableObjectProxy):
    __slots__ = ['__module_cache', '__device', '__modules', '__parameters']

    def __init__(self, wrapped, device, module_cache):
        self.__device = device
        self.__module_cache = module_cache
        self.__modules = None
        super(ModuleProxy, self).__init__(wrapped)
        # object.__setattr__(self, '__device__', device)

    def __getattr__(self, name):
        if name == '__wrapped__':
            raise ValueError('wrapper has not been initialised')

        # _parameters = self._parameters
        # if name in _parameters:
        #     return _parameters[name]
        # if '_buffers' in self.__dict__:
        #     _buffers = self.__dict__['_buffers']
        #     if name in _buffers:
        #         return _buffers[name]
        # if '_modules' in self.__dict__:
        modules = self._modules
        if name in modules:
            return modules[name]

        return getattr(self.__wrapped__, name)

    @property
    def _modules(self):
        if self.__modules is None:
            m = OrderedDict()
            for key, child in self.__wrapped__._modules.items():
                proxy = self.__module_cache.get(child)
                if proxy is None:
                    proxy = ModuleProxy(child, self.__device, self.__module_cache)
                    self.__module_cache[child] = proxy
                m[key] = proxy
            self.__modules = m
        return self.__modules

    @property
    def _foobarbaz(self):
        return '_foobarbaz'
    def callme(self):
        return 'hiya'

def replicate2(network, device_ids):
    device_ids = tuple(device_ids)
    num_replicas = len(device_ids)

    params, param_indices = parameters_and_buffers(network)
    param_copies = torch._C._functions.Broadcast(device_ids)(*params)
    param_copies = [param_copies[i:i + len(params)]
                    for i in range(0, len(param_copies), len(params))]

    modules = list(network.modules())
    module_copies = [[] for device in device_ids]
    module_indices = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module.__new__(type(module))
            replica.__dict__ = module.__dict__.copy()
            replica._parameters = replica._parameters.copy()
            replica._buffers = replica._buffers.copy()
            replica._modules = replica._modules.copy()
            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            module_idx = module_indices[child]
            for j in range(num_replicas):
                replica = module_copies[j][i]
                replica._modules[key] = module_copies[j][module_idx]
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = param_copies[j][param_idx]
        for key, buf in module._buffers.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                param_idx = param_indices[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = param_copies[j][param_idx]

    return [module_copies[j][0] for j in range(num_replicas)]


def replicate(module, device_ids):
    from ._functions import Broadcast
    seen_params = set()
    param_remap = [{} for dev_id in device_ids]
    for param in module.parameters():
        if param in seen_params:
            continue
        seen_params.add(param)
        param_copies = Broadcast(device_ids)(param)
        for param_copy, remap in zip(param_copies, param_remap):
            remap[param] = param_copy
    for m in module.modules():
        for buffer in m._buffers.values():
            copies = [buffer for i in device_ids]
            # copies = comm.broadcast(buffer, device_ids)
            for buf_copy, remap in zip(copies, param_remap):
                remap[buffer] = buf_copy
    return [_replicate_module(module, device_id, remap)
            for device_id, remap in zip(device_ids, param_remap)]
