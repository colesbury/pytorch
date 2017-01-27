#pragma once

#include <Python.h>
#include <memory>
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/autograd/native_function.h"

namespace torch { namespace autograd {

struct THPNativeFunction {
  PyObject_HEAD
  std::shared_ptr<NativeForwardFunction> cdata;
};

template<typename Ctor>
PyObject* Function_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr obj = type->tp_alloc(type, 0);
  if (!obj) return NULL;
  THPNativeFunction* f = (THPNativeFunction*)obj.get();
  new (&f->cdata) std::shared_ptr<NativeForwardFunction>(Ctor()(args));
  if (!f->cdata) {
    return NULL;
  }
  return obj.release();
}

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name);

template<typename Ctor>
PyTypeObject* createFunctionPyTypeObject(PyTypeObject& type, const char* name)
{
  type.tp_new = &Function_pynew<Ctor>;
  return _initFunctionPyTypeObject(type, name);
}

// conversion utilities for PyArg_ParseTuple
int TensorConverter(PyObject* obj, std::unique_ptr<thpp::Tensor>* address);

}} // namespace torch::autograd
