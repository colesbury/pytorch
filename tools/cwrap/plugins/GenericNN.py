import copy
from string import Template
from . import CWrapPlugin


class GenericNN(CWrapPlugin):
    INPUT_TYPE_CHECK = Template("""\
  bool is_cuda = $input->isCuda();
  auto type = $input->type();
  checkTypes(is_cuda, type, $tensor_args);
""")

    HEADER_TEMPLATE = Template("void $name($args);")

    WRAPPER_TEMPLATE = Template("""\
void $name($args)
{
$type_check
  $options
  } else {
    throw std::runtime_error("invalid arguments");
  }
}
""")

    THNN_TEMPLATE = Template("""\
    if (type == thpp::Type::FLOAT) {
        THNN_Float$name(
            NULL,
            $float_args);
    } else if (type == thpp::Type::DOUBLE) {
        THNN_Double$name(
            NULL,
            $double_args);
    } else {
        throw std::runtime_error("unsupported tensor type");
    }""")

    THCUNN_TEMPLATE = Template("""\
    if (type == thpp::Type::FLOAT) {
        THNN_Cuda$name(
            state,
            $float_args);
    } else if (type == thpp::Type::DOUBLE) {
        THNN_CudaDouble$name(
            state,
            $double_args);
    } else if (type == thpp::Type::HALF) {
        THNN_CudaHalf$name(
            state,
            $half_args);
    } else {
        throw std::runtime_error("unsupported tensor type");
    }""")

    INDEX_TENSOR_TYPES = {'THIndexTensor*', 'THCIndexTensor*'}

    REAL_TENSOR_TYPES = {'THTensor*', 'THCTensor*'}

    INPUT_ARGUMENT_MAP = {
        'THNNState*': 'void*',
        'THCState*': 'void*',
        'THTensor*': 'thpp::Tensor*',
        'THCTensor*': 'thpp::Tensor*',
        'THIndexTensor*': 'thpp::Tensor*',
        'THIndex_t': 'long',
        'real': 'double',
    }

    def __init__(self, header=False):
        self.header = header
        self.declarations = []

    def process_full_file(self, base_wrapper):
        if self.header:
            wrapper = '#pragma once\n\n'
            wrapper += '#include <THPP/Tensor.hpp>\n\n'
        else:
            wrapper = '#include "THNN_generic.h"\n'
            wrapper = '#include "THNN_generic.inc.h"\n\n'
        wrapper += 'namespace torch { namespace nn {\n\n'
        wrapper += base_wrapper
        wrapper += '}} // namespace torch::nn\n'
        return wrapper

    def process_declarations(self, declarations):
        for declaration in declarations:
            base_args = declaration['options'][0]['arguments']
            for option in declaration['options']:
                for idx, arg in enumerate(option['arguments']):
                    arg['formal_name'] = base_args[idx]['name']
                    arg['formal_type'] = base_args[idx]['type']
                    if idx != 1:
                        arg['ignore_check'] = True
        return declarations

    def get_arg_accessor(self, arg, option):
        return self.get_type_unpack(arg, option)

    def process_option_code_template(self, template, option):
        code = '// fill me in'

        def cast(arg, CReal, real):
            name = arg['formal_name']
            type = arg['type']
            if type in self.REAL_TENSOR_TYPES:
                return ('{name} ? (TH{CReal}Tensor*){name}->cdata() : NULL'
                        .format(CReal=CReal, name=name))
            elif type in self.INDEX_TENSOR_TYPES:
                return '({type}){name}->cdata()'.format(type=type, name=name)
            elif type == 'THCState*':
                return '({}){}'.format(type, name)
            elif type == 'real':
                if real == 'half':
                    return 'THC_float2half({})'.format(name)
                return '({real}){name}'.format(real=real, name=name)
            return name

        if option['backend'] == 'nn':
            float_args = []
            double_args = []
            for idx, arg in enumerate(option['arguments']):
                float_args.append(cast(arg, 'Float', 'float'))
                double_args.append(cast(arg, 'Double', 'double'))

            code = self.THNN_TEMPLATE.substitute(
                name=option['cname'],
                float_args=',\n'.join(float_args),
                double_args=',\n'.join(double_args))

        elif option['backend'] == 'cunn':
            float_args = []
            double_args = []
            half_args = []
            for idx, arg in enumerate(option['arguments']):
                float_args.append(cast(arg, 'Cuda', 'float'))
                double_args.append(cast(arg, 'CudaDouble', 'double'))
                half_args.append(cast(arg, 'CudaHalf', 'half'))

            code = self.THCUNN_TEMPLATE.substitute(
                name=option['cname'],
                float_args=',\n'.join(float_args),
                double_args=',\n'.join(double_args),
                half_args=',\n'.join(half_args))

        return [code, '']

    def get_type_unpack(self, arg, option):
        return Template(arg['name'])

    def get_type_check(self, arg, option):
        if option['backend'] == 'cunn':
            return Template('is_cuda')
        else:
            return Template('!is_cuda')

    def get_formal_args(self, arguments):
        formal_args = []
        for arg in arguments:
            arg = copy.copy(arg)
            new_type = self.INPUT_ARGUMENT_MAP.get(arg['type'])
            if new_type is not None:
                arg['type'] = new_type
            formal_args.append(arg)
        return formal_args

    def get_wrapper_template(self, declaration):
        # get formal arguments string
        base_arguments = declaration['options'][0]['arguments']
        args = self.get_formal_args(base_arguments)
        arg_str = ', '.join([arg['type'] + ' ' + arg['name'] for arg in args])

        if self.header:
            return Template(self.HEADER_TEMPLATE.safe_substitute(args=arg_str))

        checked_args = []
        for arg in base_arguments:
            if arg['type'] in self.REAL_TENSOR_TYPES:
                name = arg.get('formal_name', arg['name'])
                checked_args += ['"' + name + '"', '&' + name]
        checked_args += ['NULL']

        # check input type
        type_check = self.INPUT_TYPE_CHECK.substitute(
            input=args[1]['name'],
            tensor_args=', '.join(checked_args),)

        return Template(self.WRAPPER_TEMPLATE.safe_substitute(
            args=arg_str,
            type_check=type_check))
