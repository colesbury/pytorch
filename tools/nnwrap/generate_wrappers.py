import os
import sys
from string import Template, ascii_lowercase
from ..cwrap import cwrap
from ..cwrap.plugins import StandaloneExtension, NullableArguments, AutoGPU

BASE_PATH = os.path.realpath(os.path.join(__file__, '..', '..', '..'))
WRAPPER_PATH = os.path.join(BASE_PATH, 'torch', 'csrc', 'nn')
THNN_UTILS_PATH = os.path.join(BASE_PATH, 'torch', '_thnn', 'utils.py')

def import_module(name, path):
    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    elif sys.version_info >= (3, 0):
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader(name, path).load_module()
    else:
        import imp
        return imp.load_source(name, path)

thnn_utils = import_module('torch._thnn.utils', THNN_UTILS_PATH)

FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  cname: $cname
  arguments:
""")

COMMON_TRANSFORMS = {
    'THIndex_t': 'long',
    'THCIndex_t': 'long',
    'THInteger_t': 'int',
}
COMMON_CPU_TRANSFORMS = {
    'THNNState*': 'void*',
    'THIndexTensor*': 'THLongTensor*',
    'THIntegerTensor*': 'THIntTensor*',
}
COMMON_GPU_TRANSFORMS = {
    'THCState*': 'void*',
    'THCIndexTensor*': 'THCudaLongTensor*',
}

TYPE_TRANSFORMS = {
    'Float': {
        'THTensor*': 'THFloatTensor*',
        'real': 'float',
    },
    'Double': {
        'THTensor*': 'THDoubleTensor*',
        'real': 'double',
    },
    'CudaHalf': {
        'THCTensor*': 'THCudaHalfTensor*',
        'real': 'half',
    },
    'Cuda': {
        'THCTensor*': 'THCudaTensor*',
        'real': 'float',
    },
    'CudaDouble': {
        'THCTensor*': 'THCudaDoubleTensor*',
        'real': 'double',
    },
}
for t, transforms in TYPE_TRANSFORMS.items():
    transforms.update(COMMON_TRANSFORMS)

for t in ['Float', 'Double']:
    TYPE_TRANSFORMS[t].update(COMMON_CPU_TRANSFORMS)
for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
    TYPE_TRANSFORMS[t].update(COMMON_GPU_TRANSFORMS)


def wrap_function(name, type, arguments):
    cname = 'THNN_' + type + name
    declaration = ''
    declaration += 'extern "C" void ' + cname + '(' + ', '.join(TYPE_TRANSFORMS[type].get(arg.type, arg.type) for arg in arguments) + ');\n'
    declaration += FUNCTION_TEMPLATE.substitute(name=type + name, cname=cname)
    indent = ' ' * 4
    dict_indent = ' ' * 6
    prefix = indent + '- '
    for arg in arguments:
        if not arg.is_optional:
            declaration += prefix + TYPE_TRANSFORMS[type].get(arg.type, arg.type) + ' ' + arg.name + '\n'
        else:
            t = TYPE_TRANSFORMS[type].get(arg.type, arg.type)
            declaration += prefix + 'type: ' + t        + '\n' + \
                      dict_indent + 'name: ' + arg.name + '\n' + \
                      dict_indent + 'nullable: True' + '\n'
    declaration += ']]\n\n\n'
    return declaration

def generate_wrappers():
    wrap_nn()
    wrap_cunn()
    wrap_generic()

def wrap_nn():
    wrapper = '#include <TH/TH.h>\n\n\n'
    nn_functions = thnn_utils.parse_header(thnn_utils.THNN_H_PATH)
    for fn in nn_functions:
        for t in ['Float', 'Double']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THNN.cwrap', plugins=[
        StandaloneExtension('torch._thnn._THNN'),
        NullableArguments(),
    ])

def wrap_cunn():
    wrapper = '#include <TH/TH.h>\n'
    wrapper += '#include <THC/THC.h>\n\n\n'
    cunn_functions = thnn_utils.parse_header(thnn_utils.THCUNN_H_PATH)
    for fn in cunn_functions:
        for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THCUNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THCUNN.cwrap', plugins=[
        StandaloneExtension('torch._thnn._THCUNN'),
        NullableArguments(),
        AutoGPU(has_self=False),
    ])

GENERIC_FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  options:
""")

def wrap_generic_function(name, nn_arguments, cunn_arguments):
    cname = 'THNN_' + type + name
    declaration = ''
    indent = ' ' * 4
    dict_indent = ' ' * 6
    prefix = indent + '- '
    declaration = ''
    declaration += FUNCTION_TEMPLATE.substitute(name=name)
    for arg in arguments:
        if not arg.is_optional:
            declaration += prefix + TYPE_TRANSFORMS[type].get(arg.type, arg.type) + ' ' + arg.name + '\n'
        else:
            t = TYPE_TRANSFORMS[type].get(arg.type, arg.type)
            declaration += prefix + 'type: ' + t        + '\n' + \
                      dict_indent + 'name: ' + arg.name + '\n' + \
                      dict_indent + 'nullable: True' + '\n'
    declaration += ']]\n\n\n'
    return declaration


def wrap_generic():
    import itertools
    from collections import OrderedDict
    nn_functions = thnn_utils.parse_header(thnn_utils.THNN_H_PATH)
    cunn_functions = thnn_utils.parse_header(thnn_utils.THCUNN_H_PATH)
    functions = OrderedDict()

    def genericize_arg(arg):
        arg.type = (arg.type.replace('THNNState', 'void')
                            .replace('THCState', 'void')
                            .replace('THTensor*', 'thpp::Tensor&')
                            .replace('THCTensor*', 'thpp::Tensor&')
                            .replace('THCIndexTensor*', 'IndexTensor&')
                            .replace('THIndexTensor*', 'IndexTensor&')
                            .replace('THIndex_t', 'long'))
        return arg

    for fn in nn_functions:
        assert fn.name not in functions
        args = [genericize_arg(arg) for arg in fn.arguments]
        functions[fn.name] = {
            'name': fn.name,
            'args': args,
            'types': ['float', 'double']
        }

    for fn in cunn_functions:
        args = [genericize_arg(arg) for arg in fn.arguments]
        if fn.name in functions:
            a = [arg.type for arg in args]
            b = [arg.type for arg in functions[fn.name]['args']]
            if a != b:
                print(fn.name)
                print(a)
                print(b)
