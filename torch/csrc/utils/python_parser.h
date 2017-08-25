#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <ATen/ATen.h>

#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/python_numbers.h"

namespace torch {

enum ParameterType {
  TENSOR,
  SCALAR,
  INT64,
  DOUBLE,
  TENSOR_LIST,
  INT_LIST,
  GENERATOR,
  BOOL,
  STORAGE
};

struct FunctionParameter {
  explicit FunctionParameter(std::string fmt);

  bool check(PyObject* obj);
  void set_default_str(const std::string& str);

  ParameterType type_;
  bool optional;
  bool optional_positional;
  std::string name;
  THPObjectPtr python_name;

  at::Scalar default_scalar;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
  };
};

struct FunctionSignature {
  explicit FunctionSignature(std::string fmt);

  int num_parameters() { return params.size(); }
  std::vector<FunctionParameter>& parameters() { return params; }
  bool parse(PyObject* args, PyObject* kwargs, PyObject* dst[]);

  std::string name;
  std::vector<FunctionParameter> params;
  size_t min_args;
  size_t max_args;
};

struct PythonArgs {
  PythonArgs(int idx, const FunctionSignature& signature, PyObject** args)
    : idx(idx)
    , signature(signature)
    , args(args) {}

  int idx;
  const FunctionSignature& signature;
  PyObject** args;

  inline at::Tensor tensor(int i) {
    if (!args[i]) return at::Tensor();
    return at::Tensor(reinterpret_cast<THPVariable*>(args[i])->cdata, true);
  }

  inline at::Scalar scalar(int i) {
    if (!args[i]) return signature.params[i].default_scalar;
    if (PyFloat_Check(args[i])) {
      return at::Scalar(THPUtils_unpackDouble(args[i]));
    }
    return at::Scalar(THPUtils_unpackLong(args[i]));
  }

  inline std::vector<int64_t> intlist(int i) {
    if (!args[i]) return std::vector<int64_t>();
    PyObject* arg = args[i];
    auto tuple = PyTuple_Check(arg);
    auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
    std::vector<int64_t> res(size);
    for (int idx = 0; idx < size; idx++) {
      PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
      res[idx] = THPUtils_unpackLong(obj);
    }
    return res;
  }

  inline int64_t toInt64(int i) {
    if (!args[i]) return signature.params[i].default_int;
    return THPUtils_unpackLong(args[i]);
  }

  inline double toDouble(int i) {
    if (!args[i]) return signature.params[i].default_double;
    return THPUtils_unpackDouble(args[i]);
  }

  inline bool toBool(int i) {
    return args[i] == Py_True;
  }
};

struct PythonParser {
  explicit PythonParser(std::vector<std::string> fmts);

  PythonArgs parse(PyObject* args, PyObject* kwargs, PyObject* dst[]);

private:
  std::vector<std::string> fmts;
  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  size_t max_args;
};

} // namespace torch
