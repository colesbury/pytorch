import argparse
import os
import re
import yaml
from tools.shared.module_loader import import_module

CodeTemplate = import_module('code_template', 'torch/lib/ATen/code_template.py').CodeTemplate

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals}) override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${api_name}(${formals}) {
    ${type_definition_body}
}
""")

METHOD_DEFINITION_NYI = CodeTemplate("""\
throw std::runtime_error("${api_name}: NYI");""")

METHOD_DEFINITION_FALLTHROUGH = CodeTemplate("""\
return baseType->${api_name}(${unpacked_args});""")

METHOD_DEFINITION_FALLTHROUGH_TENSOR = CodeTemplate("""\
return Tensor(new VariableTensor(baseType->${api_name}(${args})), false);""")

UNWRAP_TENSOR = CodeTemplate("""\
auto& ${arg_name}_ = checked_unpack(${arg_name}, "${arg_name}", ${arg_pos});""")

FUNCTION_DECLARATION = CodeTemplate("""\
struct ${op} : public Function {
  using Function::Function;
  variable_list apply(const variable_list& inputs) override;
  std::string name() override { return "${op}"; }
  ${saved_variables}
};
""")

FUNCTION_DEFINITION = CodeTemplate("""\
variable_list ${op}::apply(const variable_list& inputs) {
  variable_list grad_inputs(${num_inputs});
  ${body}
  return grad_inputs;
}
""")

DERIVATIVE = CodeTemplate("""\
if (should_compute_output(${i})) {
  ${unpack_save_variables}
  grad_inputs[${i}] = ${derivative};
}
""")

METHOD_DEFINITION_DERIVATIVE = CodeTemplate("""\
auto output_ = newVariableTensor(baseType, baseType->${api_name}(${unpacked_args}));
auto flags = Function::flags({ ${tensor_args} });
output_->requires_grad = flags.is_executable;
output_->is_volatile = flags.is_volatile;
if (flags.is_executable) {
  auto grad_fn = std::make_shared<${op}>(std::move(flags));
  ${save_variables}
  output_->output_nr = grad_fn->num_inputs++;
  output_->grad_fn = grad_fn;
}
return ${return_value};
""")

PY_VARIABLE_METHOD_VARARGS = CodeTemplate("""\
static PyObject * THPVariable_${name}(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonParser parser({
    ${prototypes}
  });
  Tensor self_(reinterpret_cast<THPVariable*>(self)->cdata, true);
  PyObject* parsed_args[${max_args}];
  auto r = parser.parse(args, kwargs, parsed_args);
  ${dispatch}
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
""")

PY_VARIABLE_METHOD_DISPATCH = CodeTemplate("""\
${cond} (r.idx == ${i}) {
  return wrap(self_.${name}(${args}));
""")

PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
static PyObject * THPVariable_${name}(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor self_(reinterpret_cast<THPVariable*>(self)->cdata, true);
  return wrap(self_.${name}());
  END_HANDLE_TH_ERRORS
}
""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)THPVariable_${name}, ${flags}, NULL},""")


template_path = os.path.join(os.path.dirname(__file__), 'templates')

VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')
VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')

FUNCTIONS_H = CodeTemplate.from_file(template_path + '/Functions.h')
FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/Functions.cpp')

derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')

FALLTHROUGH_RETURN_TYPES = {'int64_t', 'void*', 'bool', 'IntList'}


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def write(dirname, name, template, env):
    env['generated_comment'] = 'generated from tools/autograd/templates/{}'.format(name)
    path = os.path.join(dirname, name)
    with open(path, 'w') as f:
        f.write(template.substitute(env))


def load_derivatives(path):
    with open(path, 'r') as f:
        definitions = yaml.load(f, Loader=Loader)

    options = []
    for defn in definitions:
        option = {}
        if '(' not in defn['name']:
            continue
        name, params = re.match('(\w+)\((.*)\)', defn['name']).groups()
        option['name'] = name
        option['args'] = []
        option['num_inputs'] = 0
        option['prototype'] = defn['name']  # with default
        option['signature'] = re.sub('=[^,\\)]*', '', defn['name'])
        option['fallthrough'] = defn.get('fallthrough', False)

        derivatives = []
        for param in params.split(', '):
            if param == '':
                continue
            arg = {}
            arg['type'], name = param.split(' ')
            if '=' in name:
                name, default = name.split('=')
                arg['optional'] = True
                arg['default'] = default
            arg['name'] = name
            option['args'].append(arg)

            if name in defn:
                arg['derivative'] = defn[name]
                derivatives.append(defn[name])
                option['num_inputs'] += 1

        for arg in option['args']:
            arg['saved'] = any(arg['name'] in d for d in derivatives)
        options.append(option)
    return options


def group_by_name(options):
    res = {}
    for option in options:
        name = option['name']
        if name not in res:
            res[name] = []
        res[name].append(option)
    return res


def create_autograd_functions(declarations):
    function_definitions = []
    function_declarations = []

    for name, options in group_by_name(declarations).items():
        for i, option in enumerate(options):
            name = option['name']
            option['op'] = name[0].upper() + name[1:] + 'Backward'
            if len(options) > 1:
                option['op'] += str(i)

    def process_function(op):
        if op['fallthrough']:
            return

        saved_variables = []
        for arg in op['args']:
            name = arg['name']
            if arg['saved']:
                if arg['type'] == 'Tensor':
                    saved_variables.append('SavedVariable {}_;'.format(name))
                elif arg['type'] == 'IntList':
                    saved_variables.append('std::vector<int64_t> {};'.format(name))
                else:
                    saved_variables.append('{} {};'.format(arg['type'], name))
        op['saved_variables'] = saved_variables

        body = []
        body.append('auto& grad = inputs[0];')

        def unpack_args(derivative):
            unpack = []
            for arg in op['args']:
                name = arg['name']
                if name in derivative and arg['type'] == 'Tensor':
                    unpack.append('auto {} = {}_.unpack();'.format(name, name))
            return unpack

        i = 0
        for arg in op['args']:
            derivative = arg.get('derivative')
            if derivative is None:
                continue
            body.append(DERIVATIVE.substitute({
                'unpack_save_variables': unpack_args(derivative),
                'i': i,
                'derivative': derivative,
            }))
            i += 1

        op['body'] = body
        function_declarations.append(FUNCTION_DECLARATION.substitute(op))
        function_definitions.append(FUNCTION_DEFINITION.substitute(op))

    for option in declarations:
        process_function(option)

    return function_declarations, function_definitions


def create_variable_type(aten_declarations):
    type_declarations = []
    type_definitions = []

    def save_variables(option, derivative):
        stmts = []
        for arg in derivative['args']:
            if not arg['saved']:
                continue
            save_name = arg['name']
            expr = arg['name']
            if arg['type'] == 'Tensor':
                save_name += '_'
                expr = 'SavedVariable({}, nullptr)'.format(arg['name'])
            stmts.append('grad_fn->{} = {};'.format(save_name, expr))
        return stmts

    def process_function(option):
        option['formals'] = [arg['type'] + ' ' + arg['name']
                             for arg in option['arguments']]
        option['args'] = [arg['name'] for arg in option['arguments']]
        option['api_name'] = option['name']
        return_type = format_return_type(option['returns'])
        option['return_type'] = return_type
        if 'derivative' in option:
            derivative = option['derivative']
            option['op'] = derivative['op']
            option['save_variables'] = save_variables(option, derivative)
            option['tensor_args'] = [arg['name'] for arg in option['arguments']
                                     if arg['dynamic_type'] == 'Tensor']
        if return_type == 'Scalar':
            option['return_value'] = 'Scalar(Tensor(output_, false))'
        else:
            option['return_value'] = 'Tensor(output_, false)'

        option['type_definition_body'] = emit_body(option)

        type_declarations.append(METHOD_DECLARATION.substitute(option))

        if option['name'] != 'm_contiguous':
            type_definitions.append(METHOD_DEFINITION.substitute(option))

    def unpack_args(option):
        body = []
        unpacked_args = []
        for i, arg in enumerate(option['arguments']):
            if arg['dynamic_type'] == 'Tensor':
                env = {'arg_name': arg['name'], 'arg_pos': i}
                body.append(UNWRAP_TENSOR.substitute(env))
                unpacked_args.append(arg['name'] + '_')
            else:
                unpacked_args.append(arg['name'])
        option['unpacked_args'] = unpacked_args
        return body

    def emit_body(option):
        if len(option['returns']) != 1:
            return METHOD_DEFINITION_NYI.substitute(option)

        body = []
        body += unpack_args(option)
        if 'derivative' in option:
            body.extend(METHOD_DEFINITION_DERIVATIVE.substitute(option).split('\n'))
            return body

        if option['return_type'] in FALLTHROUGH_RETURN_TYPES:
            body.extend(METHOD_DEFINITION_FALLTHROUGH.substitute(option).split('\n'))
            return body

        return METHOD_DEFINITION_NYI.substitute(option)

    for function in aten_declarations:
        process_function(function)

    return type_declarations, type_definitions


def create_python_bindings(aten_decls, derivatives):
    py_methods = []
    py_method_defs = []
    unpack_methods = {
        'int64_t': 'toInt64',
        'bool': 'toBool'
    }

    def group_by_name(derivatives):
        res = {}
        for option in derivatives:
            name = option['name']
            if '(' not in option['signature']:
                continue
            if name not in res:
                res[name] = []
            res[name].append(option)
        return res

    def emit_dispatch(i, option):
        env = {}
        args = []
        params = [arg for arg in option['args'] if arg['name'] != 'self']
        for arg_idx, arg in enumerate(params):
            unpack = unpack_methods.get(arg['type'], arg['type'].lower())
            args.append('r.{}({})'.format(unpack, arg_idx))
        env['i'] = i
        env['name'] = option['name']
        env['args'] = args
        env['cond'] = 'if' if i == 0 else '} else if'
        return PY_VARIABLE_METHOD_DISPATCH.substitute(env)

    def process_option(name, options):
        env = {}
        env['name'] = name
        env['prototypes'] = []
        env['max_args'] = max(len(d['args']) for d in options)
        for d in options:
            prototype = d['prototype']
            prototype = prototype.replace('Tensor self, ', '')
            prototype = prototype.replace(', Tensor self', '')
            prototype = prototype.replace('Tensor self', '')
            env['prototypes'].append('"{}",'.format(prototype))
        env['dispatch'] = []
        env['flags'] = 'METH_VARARGS | METH_KEYWORDS'

        dispatch = []
        for i, option in enumerate(options):
            dispatch.append(emit_dispatch(i, option))
        dispatch.append('}')
        env['dispatch'] = dispatch

        if len(options) == 1 and len(options[0]['args']) == 1:
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['flags'] = 'METH_NOARGS'
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'

        py_methods.append(tmpl.substitute(env))
        py_method_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))

    tmp = group_by_name(derivatives)
    for name in sorted(tmp.keys()):
        process_option(name, tmp[name])

    return py_methods, py_method_defs


def match_derivatives(aten_decls, derivative_decls):
    derivative_signatures = {d['signature']: d for d in derivative_decls}

    def process_option(option):
        args = []
        for arg in option['arguments']:
            simple_type = arg['type'].replace(' &', '').replace('const ', '')
            args.append(simple_type + ' ' + arg['name'])
        signature = '{}({})'.format(option['name'], ', '.join(args))
        option['signature'] = signature
        if signature in derivative_signatures:
            option['derivative'] = derivative_signatures[signature]

    for option in aten_decls:
        process_option(option)


def gen_variable_type(declarations, out):
    with open(declarations, 'r') as f:
        aten_decls = yaml.load(f, Loader=Loader)
        #aten_decls = [opt for opt in aten_decls if opt['method_of'] == 'Type']
        for opt in aten_decls:
             opt['name'] = opt['method_prefix'] + opt['name']

    derivatives = load_derivatives(derivatives_path)
    match_derivatives(aten_decls, derivatives)

    function_decls, function_defs = create_autograd_functions(derivatives)
    type_declarations, type_definitions = create_variable_type(aten_decls)
    py_methods, py_method_defs = create_python_bindings(aten_decls, derivatives)

    env = {}
    env['autograd_function_declarations'] = function_decls
    env['autograd_function_definitions'] = function_defs
    env['type_derived_method_declarations'] = type_declarations
    env['type_derived_method_definitions'] = type_definitions
    env['py_methods'] = py_methods
    env['py_method_defs'] = py_method_defs

    if not os.path.exists(out):
        os.makedirs(out)
    write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    write(out, 'VariableType.cpp', VARIABLE_TYPE_CPP, env)
    write(out, 'python_variable_methods.cpp', VARIABLE_METHODS_CPP, env)
    write(out, 'Functions.h', FUNCTIONS_H, env)
    write(out, 'Functions.cpp', FUNCTIONS_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_variable_type(args.declarations, args.out)


if __name__ == '__main__':
    main()
