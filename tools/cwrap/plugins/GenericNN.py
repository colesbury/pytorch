import copy
import os
from string import Template
from . import CWrapPlugin


class GenericNN(CWrapPlugin):
    TYPE_UNPACK = {
    }

    TYPE_CHECK = {
    }

    OPTION_TEMPLATE = Template("""skip""")

    INPUT_TYPE_CHECK = Template("""\
  bool is_cuda = $input.isCuda();
  auto type = $input.type();
""")

    TYPE_CHECK = Template("""\
checkTypes(is_cuda, type, $args);""")

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

    TYPE_NAMES = {
    }

    REAL_TENSOR_TYPES = {
        'THTensor*',
        'THCTensor*',
    }

    INPUT_ARGUMENT_MAP = {
        'THNNState*': 'void*',
        'THCState*': 'void*',
        'THTensor*': 'thpp::Tensor&',
        'THCTensor*': 'thpp::Tensor&',
        'THIndexTensor*': 'thpp::Tensor&',
        'THIndex_t': 'long',
        'real': 'double',
    }

    def __init__(self, module_name):
        self.module_name = module_name
        self.declarations = []

    def process_declarations(self, declarations):
        for declaration in declarations:
            base_args = declaration['options'][0]['arguments']
            for option in declaration['options']:
                for idx, arg in enumerate(option['arguments']):
                    formal_name = base_args[idx]['name']
                    if formal_name != arg['name']:
                        arg['formal_name'] = formal_name
                    if idx != 1:
                        arg['ignore_check'] = True
        return declarations

    # def process_full_file(self, code):
    #     short_name = self.module_name.split('.')[-1]
    #     new_code = MODULE_HEAD
    #     new_code += code
    #     new_code += self.declare_module_methods()
    #     new_code += MODULE_TAIL.substitute(full_name=self.module_name, short_name=short_name)
    #     return new_code

    # def process_wrapper(self, code, declaration):
    #     self.declarations.append(declaration)
    #     return code
    #
    # def declare_module_methods(self):
    #     module_methods = ''
    #     for declaration in self.declarations:
    #         module_methods += REGISTER_METHOD_TEMPLATE.substitute(name=declaration['name'])
    #     return MODULE_METHODS_TEMPLATE.substitute(METHODS=module_methods)
    #
    def get_arg_accessor(self, arg, option):
        return self.get_type_unpack(arg, option)

    def process_option_code_template(self, template, option):
        checked_args = []
        for arg in option['arguments']:
            if arg['type'] in self.REAL_TENSOR_TYPES:
                name = arg.get('formal_name', arg['name'])
                checked_args += ['"' + name + '"', '&' + name]
        checked_args += ['NULL']
        tmpl = self.TYPE_CHECK.substitute(args=', '.join(checked_args))
        # print(template)
        print(checked_args)
        # print(option)
        return [tmpl, '']

    def get_type_unpack(self, arg, option):
        return Template(arg['name'])
        # return self.TYPE_UNPACK.get(arg['type'], None)
    #

    def get_type_check(self, arg, option):
        if option['backend'] == 'cunn':
            return Template('is_cuda')
        else:
            return Template('!is_cuda')

    # def get_return_wrapper(self, option):
    #     return Template('return;')

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

        # unpack input type
        type_check = self.INPUT_TYPE_CHECK.substitute(input=args[1]['name'])

        return Template(self.WRAPPER_TEMPLATE.safe_substitute(
            args=arg_str,
            type_check=type_check))
