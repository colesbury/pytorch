#include <Python.h>
#include "THP.h"

#include <THPP/THPP.h>
#include "torch/csrc/nn/THNN_generic.h"

using thpp::Tensor;

namespace torch { namespace autograd {

auto NativeForwardFunction::flags(const variable_list& inputs) -> FunctionFlags {
  bool requires_grad = false;
  bool is_volatile = false;
  int size = inputs.size();
  function_list prev(size);
  for (int i = 0; i != size; ++i) {
    auto& var = inputs[i];
    requires_grad |= var->requires_grad;
    is_volatile |= var->is_volatile;
    prev[i] = std::make_pair<>(var->creator, i);
  }
  requires_grad &= !is_volatile;
  return FunctionFlags(requires_grad, is_volatile, std::move(prev));
}

auto NativeForwardFunction::pythonObject() -> PyObject* {
  bool requires_grad = false;
  bool is_volatile = false;
  int size = inputs.size();
  function_list prev(size);
  for (int i = 0; i != size; ++i) {
    auto& var = inputs[i];
    requires_grad |= var->requires_grad;
    is_volatile |= var->is_volatile;
    prev[i] = std::make_pair<>(var->creator, i);
  }
  requires_grad &= !is_volatile;
  return FunctionFlags(requires_grad, is_volatile, std::move(prev));
}

}} // namespace torch::autograd
