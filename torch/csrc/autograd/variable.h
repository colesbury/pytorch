#pragma once

#include <Python.h>
#include <mutex>
#include <memory>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/VariableTensor.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

inline at::Tensor & data(const at::Tensor & var) {
  auto pImpl = dynamic_cast<VariableTensor*>(var.get());
  if (!pImpl) {
    throw std::runtime_error("expected VariableTensor");
  }
  return pImpl->data;
}

inline at::Tensor data_opt(const at::Tensor & var) {
  if (!var.defined()) {
    return at::Tensor();
  }
  auto pImpl = dynamic_cast<VariableTensor*>(var.get());
  if (!pImpl) {
    throw std::runtime_error("expected VariableTensor");
  }
  return pImpl->data;
}

// struct Variable : std::enable_shared_from_this<Variable> {
//   // WARNING: this registers the Variable as a new output
//   Variable(
//       at::Tensor data,
//       std::shared_ptr<Function> grad_fn);
//
//   Variable(
//       at::Tensor data,
//       bool requires_grad,
//       bool is_volatile);
//
//   std::shared_ptr<Function> get_grad_accumulator();
//
//   SavedVariable save(Function* saved_for);
//
//   SavedVariable save_opt(Variable* var, Function* saved_for);
//
//   // TODO: should be at::Tensor&& if we are taking ownership?
//   static inline std::shared_ptr<Variable> of(at::Tensor data, bool is_volatile=false) {
//     if (!data.defined()) {
//       return std::shared_ptr<Variable>();
//     }
//     return std::make_shared<Variable>(data, false, is_volatile);
//   }
//
//   at::Tensor data;
//   std::shared_ptr<Function> grad_fn;
//   std::shared_ptr<Variable> grad;
//   std::unique_ptr<VariableVersion> version_counter;
//   std::vector<std::shared_ptr<FunctionPreHook>> hooks;
//   std::weak_ptr<Function> grad_accumulator;
//   std::mutex grad_accumulator_lock;
//   bool requires_grad;
//   bool is_volatile;
//   // The "output number" of this variable; e.g., if this variable
//   // was the second output of a function, then output_nr == 1.
//   // We use this to make sure we can setup the backwards trace
//   // correctly when this variable is passed to another function.
//   int output_nr;
//   PyObject *pyobj;  // weak reference
// };

}} // namespace torch::autograd
