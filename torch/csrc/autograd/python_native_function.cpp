#include "python_native_function.h"

#include <Python.h>
#include <memory>
#include <stdio.h>
#include <typeindex>
#include <unordered_map>
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
    if (arg == Py_None) {
      continue;
    }
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
    output = ((THPNativeFunction*)self)->cdata->apply(vars);
    Py_BLOCK_THREADS;
  } catch (...) {
    if (_save) {
      Py_BLOCK_THREADS;
    }
    throw;
  }
  END_HANDLE_TH_ERRORS

  int num_outputs = output.size();
  if (num_outputs == 1) {
    // assume we want to unpack one element tuples for now
    return THPVariable_Wrap(output[0]);
  }

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

static PyTypeObject NativeFunctionClass;

void initNativeFunction()
{
  _initFunctionPyTypeObject(NativeFunctionClass, "torch.autograd.NativeFunction");
}

static std::unordered_map<std::type_index, THPObjectPtr> native_function_types;

PyObject* functionToPyObject(std::shared_ptr<Function> cdata)
{
  auto pfw = dynamic_cast<PyFunctionWrapper*>(cdata.get());
  if (pfw) {
    PyObject* obj = pfw->pyobj.get();
    Py_INCREF(obj);
    return obj;
  }

  auto it = native_function_types.find(std::type_index(typeid(*cdata)));
  if (it != native_function_types.end()) {
    PyTypeObject* type = (PyTypeObject*)it->second.get();
    THPObjectPtr obj = type->tp_alloc(type, 0);
    if (!obj) return NULL;
    THPNativeFunction* f = (THPNativeFunction*)obj.get();
    new (&f->cdata) std::shared_ptr<Function>(cdata);
    if (!f->cdata) {
      return NULL;
    }
    return obj.release();
  }

  return PyErr_Format(PyExc_TypeError,
      "Don't know how to create Python object for %s", typeid(*cdata).name());
}

void registerNativeFunction(const std::type_info& type, PyTypeObject* pytype)
{
  Py_INCREF((PyObject*)pytype);
  native_function_types[std::type_index(type)] = THPObjectPtr((PyObject*)pytype);
}

}} // namespace torch::autograd
