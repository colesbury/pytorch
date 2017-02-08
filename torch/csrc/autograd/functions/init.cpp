#include <Python.h>
#include "batch_normalization.h"
#include "convolution.h"
#include "broadcast.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/utils/tuple_parser.h"

using namespace torch::autograd;
using torch::TupleParser;

static PyTypeObject BatchNormClass;
static PyTypeObject BatchNormBackwardClass;

struct BatchNormCtor {
  BatchNormForward* operator()(PyObject* args) {
    std::unique_ptr<thpp::Tensor> running_mean;
    std::unique_ptr<thpp::Tensor> running_var;
    bool training;
    double momentum;
    double eps;

    TupleParser parser(args, 5);
    parser.parse(running_mean);
    parser.parse(running_var);
    parser.parse(training);
    parser.parse(momentum);
    parser.parse(eps);

    return new BatchNormForward(
        std::move(running_mean), std::move(running_var), training, momentum, eps);
  }
};

struct ConvCtor {
  ConvForward* operator()(PyObject* args) {
    ConvParams params;

    TupleParser parser(args, 7);
    parser.parse(params.stride);
    parser.parse(params.padding);
    parser.parse(params.dilation);
    parser.parse(params.transposed);
    parser.parse(params.output_padding);
    parser.parse(params.groups);
    parser.parse(params.benchmark);

    return new ConvForward(std::move(params));
  }
};

struct BroadcastCtor {
  Broadcast* operator()(PyObject* args) {
    std::vector<int> devices;

    TupleParser parser(args, 1);
    parser.parse(devices);

    return new Broadcast(devices);
  }
};

struct NoCtor {
  Function* operator()(PyObject* args) {
    throw std::runtime_error("Cannot construct");
  }
};

template<typename C, typename T>
static void addClass(PyObject* module, PyTypeObject& type, const char* name)
{
  createForwardFunctionPyTypeObject<T>(type, name);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

bool THPAutograd_initFunctions(PyObject* _unused)
{
  THPObjectPtr module = PyModule_New("torch._C._functions");
  if (!module) return false;

  addClass<BatchNormForward, BatchNormCtor>(module, BatchNormClass, "BatchNorm");
  addClass<BatchNormBackward, NoCtor>(module, BatchNormBackwardClass, "BatchNormBackward");

  static PyTypeObject ConvClass, ConvBackwardClass;
  addClass<ConvForward, ConvCtor>(module, ConvClass, "Conv");
  addClass<ConvBackward, NoCtor>(module, ConvBackwardClass, "BatchNormBackward");

  static PyTypeObject BroadcastClass;
  addClass<Broadcast, BroadcastCtor>(module, BroadcastClass, "Broadcast");

  THPObjectPtr parent = PyImport_ImportModule("torch._C");
  if (!parent) return false;
  PyModule_AddObject(parent.get(), "_functions", module.release());
  return true;
}
