#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct FunctionFlags {
  FunctionFlags(
      bool requires_grad,
      bool is_volatile,
      function_list previous_functions) noexcept
    : requires_grad(requires_grad)
    , is_volatile(is_volatile)
    , previous_functions(std::move(previous_functions)) {}
  FunctionFlags(const FunctionFlags&) = delete;
  FunctionFlags(FunctionFlags&&) = default;

  bool requires_grad;
  bool is_volatile;
  function_list previous_functions;
};

struct NativeFunction : public Function {
  NativeFunction()
    : num_outputs(0)
    , previous_functions()
    , requires_grad(false)
    , is_volatile(false)
    , is_stochastic(false)
    , python_object(nullptr)
    {}

  NativeFunction(FunctionFlags flags)
    : num_outputs(0)
    , previous_functions(std::move(flags.previous_functions))
    , requires_grad(flags.requires_grad)
    , is_volatile(flags.is_volatile)
    , is_stochastic(false)
    , python_object(nullptr)
    {}

  virtual variable_list apply(const variable_list& gradOutputs) = 0;

  static inline SavedVariable save_optional(THVariable* var) {
    return var ? var->save() : SavedVariable();
  }
  static FunctionFlags flags(const variable_list& inputs);

  virtual function_list previousFunctions() override {
     return previous_functions;
  }
  virtual int numOutputs() const override { return num_outputs; }
  virtual bool requiresGrad() const override { return requires_grad; }
  virtual bool isStochastic() const override { return is_stochastic; }

  int num_outputs;
  function_list previous_functions;
  bool requires_grad;
  bool is_volatile;
  bool is_stochastic;
  PyObject* python_object; // weak reference
};

}} // namespace torch::autograd
