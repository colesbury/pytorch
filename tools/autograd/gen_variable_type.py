import argparse
import os
import yaml
from tools.shared.module_loader import import_module

CodeTemplate = import_module('code_template', 'torch/lib/ATen/code_template.py').CodeTemplate

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


TYPE_METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals}) override;
""")

TYPE_METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${api_name}(${formals}) {
    ${type_definition_body}
}
""")

TYPE_METHOD_DEFINITION_NYI = """\
throw std::runtime_error("NYI");"""

TYPE_METHOD_DEFINITION_FALLTHROUGH = CodeTemplate("""\
return baseType->${api_name}(${args});""")

UNWRAP_TENSOR = CodeTemplate("""\
checked_unpack(${arg_name}, "${arg_name}", ${arg_pos})""")

template_path = os.path.join(os.path.dirname(__file__), 'templates')

VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')

FALLTHROUGH_RETURN_TYPES = {'int64_t', 'void*', 'bool'}


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def write(dirname, name, template, env):
    path = os.path.join(dirname, name)
    with open(path, 'w') as f:
        f.write(template.substitute(env))


def gen_variable_type(declarations, out):
    with open(declarations, 'r') as f:
        decls = yaml.load(f, Loader=Loader)

    def method_env(decl):
        formals = [arg['type'] + ' ' + arg['name'] for arg in decl['arguments']]

        env = {}
        env['formals'] = formals
        env['api_name'] = decl['name']
        env['return_type'] = format_return_type(decl['returns'])

        return env

    def emit_body(env, option):
        if len(option['returns']) != 1:
            return TYPE_METHOD_DEFINITION_NYI

        return_type = option['returns'][0]['type']
        if return_type not in FALLTHROUGH_RETURN_TYPES:
            return TYPE_METHOD_DEFINITION_NYI

        body = []

        checked_args = []
        for i, arg in enumerate(option['arguments']):
            if arg['dynamic_type'] == 'Tensor':
                unwrap = UNWRAP_TENSOR.substitute(
                    arg_name=arg['name'], arg_pos=i)
                body.append('auto& {}_ = {};'.format(
                    arg['name'], unwrap))
                checked_args.append(arg['name'] + '_')
            else:
                checked_args.append(arg['name'])

        body.append(TYPE_METHOD_DEFINITION_FALLTHROUGH.substitute(
            env, args=checked_args))

        return body

    declarations = []
    definitions = []
    for decl in decls:
        if decl['method_of'] == 'Type':
            env = method_env(decl)
            env['type_definition_body'] = emit_body(env, decl)

            declarations.append(TYPE_METHOD_DECLARATION.substitute(env))
            definitions.append(TYPE_METHOD_DEFINITION.substitute(env))

    write(out, 'VariableType.h', VARIABLE_TYPE_H, {
        'type_derived_method_declarations': '  '.join(declarations),
    })

    write(out, 'VariableType.cpp', VARIABLE_TYPE_CPP, {
        'type_derived_method_definitions': ''.join(definitions),
    })


def main():
    parser = argparse.ArgumentParser(
        description='Generate VariableType C++ script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_variable_type(args.declarations, args.out)


if __name__ == '__main__':
    main()
