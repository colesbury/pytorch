#include "torch/csrc/autograd/saved_variable.h"

using namespace at;

namespace torch { namespace autograd {

SavedVariable::SavedVariable(Tensor variable, Function* saved_for)
  : SavedVariable() {
  if (variable.defined()) {
    auto pImpl = dynamic_cast<VariableTensor*>(variable.get());
    if (!pImpl) {
      throw std::runtime_error("SavedVariable(): expected VariableTensor");
    }
    data = pImpl->data;
    has_grad_fn = pImpl->grad_fn != nullptr;
    grad_accumulator = pImpl->grad_accumulator;
    version = pImpl->version_counter->new_saved_ref();
    requires_grad = pImpl->requires_grad;
    is_volatile = pImpl->is_volatile;
    expected_version = **pImpl->version_counter;
    if (pImpl->grad_fn.get() != saved_for) {
      grad_fn = pImpl->grad_fn;
    }
  }
}

auto SavedVariable::unpack(std::shared_ptr<Function> saved_for) -> Tensor {
  if (!data.defined()) {
    if (version) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Tensor();
  }

  int current_version = **version;
  if (expected_version != current_version) {
    throw std::runtime_error("one of the variables "
        "needed for gradient computation has been modified by an "
        "inplace operation");
  }

  auto new_var = new VariableTensor(data);
  Tensor tensor(new_var, false);

  new_var->requires_grad = requires_grad;
  new_var->is_volatile = is_volatile;
  if (has_grad_fn && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    new_var->grad_fn = saved_for;
  } else {
    new_var->grad_fn = grad_fn;
  }
  new_var->version_counter->join_with(*version);
  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the graph).
  if (requires_grad && !new_var->grad_fn && grad_accumulator.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  new_var->grad_accumulator = grad_accumulator;

  return tensor;
}

auto SavedVariable::unpack_data(std::shared_ptr<Function> saved_for) -> Tensor {
  auto var = unpack(saved_for);
  if (var.defined()) {
    return dynamic_cast<VariableTensor*>(var.get())->data;
  }
  return Tensor();
}


const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

}} // namespace torch::autograd
