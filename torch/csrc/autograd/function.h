#ifndef THP_FUNCTION_H
#define THP_FUNCTION_H

#include <vector>
#include <memory>
#include "torch/csrc/utils/object_ptr.h"

struct THPFunction;

class python_error : public std::exception {};

struct THPVariableVersion;
struct THVariable;

namespace thpp {
  struct Tensor;
}

namespace torch { namespace autograd {

struct Function;

using tensor_list = std::vector<std::unique_ptr<thpp::Tensor>>;
using variable_list = std::vector<std::shared_ptr<THVariable>>;
using function_list = std::vector<std::pair<std::shared_ptr<Function>, int>>;

struct Function {
  Function() {};
  Function(const Function& other) = delete;
  Function(Function&& other) = delete;
  virtual ~Function() {};

  virtual variable_list apply(const variable_list& inputs) = 0;

  virtual inline void releaseVariables() {}
  virtual function_list previousFunctions() = 0;
  virtual int numOutputs() const = 0;
  virtual bool requiresGrad() const = 0;
  virtual bool isStochastic() const = 0;
};

struct PyFunctionWrapper : public Function {
  PyFunctionWrapper(PyObject *obj);
  virtual ~PyFunctionWrapper();

  virtual variable_list apply(const variable_list& inputs) override;
  virtual void releaseVariables() override;

  virtual function_list previousFunctions() override;
  virtual int numOutputs() const override;
  virtual bool requiresGrad() const override;
  virtual bool isStochastic() const override;

  THPObjectPtr pyobj;
};


}} // namespace torch::autograd

#endif
