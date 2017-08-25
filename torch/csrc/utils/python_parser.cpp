#include "torch/csrc/utils/python_parser.h"

#include <stdexcept>
#include <unordered_map>

using namespace at;

namespace torch {

static std::unordered_map<std::string, ParameterType> type_map = {
  {"Tensor", TENSOR},
  {"Scalar", SCALAR},
  {"int64_t", INT64},
  {"double", DOUBLE},
  {"TensorList", TENSOR_LIST},
  {"IntList", INT_LIST},
  {"Generator", GENERATOR},
  {"bool", BOOL},
  {"Storage", STORAGE}
};

FunctionParameter::FunctionParameter(std::string fmt)
  : optional(false)
  , optional_positional(false)
  , default_scalar(0)
{
  auto space = fmt.find(' ');
  if (space == std::string::npos) {
    throw std::runtime_error("FunctionParameter(): missing type: " + fmt);
  }

  auto type_str = fmt.substr(0, space);
  auto name_str = fmt.substr(space + 1);
  type_ = type_map[type_str];

  auto eq = name_str.find('=');
  if (eq != std::string::npos) {
    name = name_str.substr(0, eq);
    optional = true;
    set_default_str(name_str.substr(eq + 1));
  } else {
    name = name_str;
  }
#if PY_MAJOR_VERSION == 2
  python_name = PyString_InternFromString(name.c_str());
#else
  python_name = PyUnicode_InternFromString(name.c_str());
#endif
}

bool FunctionParameter::check(PyObject* obj) {
  switch (type_) {
    case TENSOR: return THPVariable_Check(obj);
    case SCALAR: return THPUtils_checkDouble(obj);
    case INT64: return THPUtils_checkLong(obj);
    case DOUBLE: return THPUtils_checkDouble(obj);
    case TENSOR_LIST: return PyTuple_Check(obj) || PyList_Check(obj);
    case INT_LIST: return PyTuple_Check(obj) || PyList_Check(obj);
    case GENERATOR: return false;
    case BOOL: return PyBool_Check(obj);
    case STORAGE: return false;
    default: throw std::runtime_error("unknown parameter type");
  }
}

void FunctionParameter::set_default_str(const std::string& str) {
  if (type_ == TENSOR) {
    if (str != "None") {
      throw std::runtime_error("default value for Tensor must be none, got: " + str);
    }
    return;
  } else if (type_ == INT64) {
    default_int = atol(str.c_str());
  } else if (type_ == BOOL) {
    default_bool = (str == "True" || str == "true");
  } else if (type_ == DOUBLE) {
    default_double = atof(str.c_str());
  } else if (type_ == SCALAR) {
    default_scalar = Scalar(atof(str.c_str()));
  }
}

FunctionSignature::FunctionSignature(std::string fmt)
  : min_args(0)
  , max_args(0)
{
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);

  auto last_offset = open_paren + 1;
  auto next_offset = last_offset;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    if (offset == std::string::npos) {
      offset = fmt.find(")", last_offset);
      done = true;
    } else {
      next_offset = offset + 2;
    }
    if (offset == std::string::npos) {
      throw std::runtime_error("missing closing parenthesis: " + fmt);
    }
    if (offset == last_offset) {
      break;
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);
    params.emplace_back(param_str);
    last_offset = next_offset;
  }

  max_args = params.size();

  // count the number of non-optional args
  for (auto& param : params) {
    if (!param.optional) {
      min_args++;
    }
  }

  // mark optional args that come before non-optional args
  // e.g. "beta" and "alpha" in:
  // addbmm(Scalar beta=1, Tensor mat, Scalar alpha=1, Tensor batch1, Tensor batch2)
  size_t positional_args = 0;
  for (auto& param : params) {
    if (param.optional) {
      param.optional_positional = positional_args < min_args;
    } else {
      positional_args++;
    }
  }
}

bool FunctionSignature::parse(PyObject* args, PyObject* kwargs, PyObject* dst[]) {
  auto nargs = PyTuple_GET_SIZE(args);
  Py_ssize_t remaining_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  Py_ssize_t arg_pos = 0;

  int i = 0;
  for (auto& param : params) {
    PyObject* obj = nullptr;
    if (arg_pos < nargs) {
      obj = PyTuple_GET_ITEM(args, arg_pos);
      if (param.check(obj)) {
        dst[i++] = obj;
        arg_pos++;
        continue;
      } else if (!param.optional_positional) {
        return false;
      }
      // fallthrough to kwarg handling
    }

    obj = kwargs ? PyDict_GetItem(kwargs, param.python_name.get()) : nullptr;
    if (obj) {
      remaining_kwargs--;
      if (!param.check(obj)) {
        return false;
      }
      dst[i++] = obj;
    } else if (param.optional) {
      dst[i++] = nullptr;
    } else {
      return false;
    }
  }

  if (remaining_kwargs > 0) {
    return false;
  }

  return true;
}

PythonParser::PythonParser(std::vector<std::string> fmts)
 : max_args(0)
{
  for (auto& fmt : fmts) {
    signatures_.push_back(FunctionSignature(fmt));
  }
  for (auto& signature : signatures_) {
    if (signature.max_args > max_args) {
      max_args = signature.max_args;
    }
  }
}

PythonArgs PythonParser::parse(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {
  int i = 0;
  for (auto& signature : signatures_) {
    if (signature.parse(args, kwargs, parsed_args)) {
      return PythonArgs(i, signature, parsed_args);
    }
    i++;
  }

  // TODO: error message
  throw std::runtime_error("bad call");
}


} // namespace torch
