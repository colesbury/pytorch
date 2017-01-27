#include <Python.h>
#include "batch_normalization.h"
#include "torch/csrc/autograd/python_native_function.h"

using namespace torch::autograd;

static PyTypeObject BatchNormClass;

struct BatchNormCtor {
  BatchNormForward* operator()(PyObject* args) {
    std::unique_ptr<thpp::Tensor> running_mean;
    std::unique_ptr<thpp::Tensor> running_var;
    char training;
    double momentum;
    double eps;

    if (!PyArg_ParseTuple(args, "O&O&Bdd:BatchNorm",
          TensorConverter, &running_mean,
          TensorConverter, &running_var,
          &training, &momentum, &eps)) {
      return NULL;
    }

    return new BatchNormForward(
        std::move(running_mean),
        std::move(running_var),
        (bool)training,
        momentum,
        eps);
  }
};

void initAutogradFunctions()
{
  THPObjectPtr module = PyImport_ImportModule("torch.nn._functions.thnn");
  createFunctionPyTypeObject<BatchNormCtor>(BatchNormClass, "BatchNorm");
  Py_INCREF(&BatchNormClass);
  PyModule_AddObject(module.get(), "BatchNorm", (PyObject*)&BatchNormClass);
}
