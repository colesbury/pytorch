// ${generated_comment}

#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_parser.h"

using at::Tensor;
using at::Scalar;

namespace torch { namespace autograd {

namespace {

inline PyObject* wrap(Tensor tensor) {
  return THPVariable_Wrap(tensor);
}

inline PyObject* wrap(bool value) {
  if (value) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

inline PyObject* wrap(int64_t value) {
  return PyLong_FromLongLong(value);
}

inline PyObject* wrap(Scalar scalar) {
  return wrap(scalar.toTensor());
}

} // anonymous namespace

${py_methods}

PyMethodDef variable_methods[] = {
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
