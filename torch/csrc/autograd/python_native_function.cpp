#include "python_native_function.h"

#include <Python.h>
#include <memory>
#include <stdio.h>
#include <THPP/THPP.h>
#include "torch/csrc/autograd/native_function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"

using namespace torch::autograd;

namespace torch { namespace autograd {

namespace {

PyObject* THPNativeFunction_call(PyObject* self, PyObject* args, PyObject *kwargs)
{
  int num_inputs = PyTuple_GET_SIZE(args);
  variable_list vars(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    if (!THPVariable_Check(arg)) {
      return PyErr_Format(PyExc_TypeError, "argument %d is not a Variable", i);
    }
    vars[i] = ((THPVariable*)arg)->cdata;
  }

  variable_list output;

  HANDLE_TH_ERRORS
  PyThreadState *_save = NULL;
  try {
    Py_UNBLOCK_THREADS;
    output = ((THPNativeFunction*)self)->cdata->forward(vars);
    Py_BLOCK_THREADS;
  } catch (...) {
    if (_save) {
      Py_BLOCK_THREADS;
    }
    throw;
  }
  END_HANDLE_TH_ERRORS

  int num_outputs = output.size();
  THPObjectPtr tuple = PyTuple_New(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyTuple_SET_ITEM(tuple.get(), i, THPVariable_Wrap(output[i]));
  }
  return tuple.release();
}

} // namespace

int TensorConverter(PyObject* obj, std::unique_ptr<thpp::Tensor>* address)
{
  try {
    *address = createTensor(obj);
  } catch (std::exception& e) {
    PyErr_Format(PyExc_TypeError,
        "expected torch.Tensor, got %s", Py_TYPE(obj)->tp_name);
    return 0;
  }
  return 1;
}

static struct PyMethodDef methods[] = {
  // {(char*)"__call__", (PyCFunction)THPNativeFunction_call, METH_VARARGS, NULL},
  {NULL}
};

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name)
{
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_basicsize = sizeof(THPNativeFunction);
  type.tp_methods = methods;
  type.tp_call = THPNativeFunction_call;
  if (PyType_Ready(&type) < 0) {
    auto msg = std::string("Unable to instantiate PyTypeObject for ") + name;
    throw std::runtime_error(msg);
  }
  return &type;
}

}} // namespace torch::autograd
