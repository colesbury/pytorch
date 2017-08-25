#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/utils/auto_gpu.h"

using namespace torch;

namespace torch { namespace autograd {

Variable::Variable(
  at::Tensor data,
  bool requires_grad,
  bool is_volatile)
    : data(data)
    , grad_fn(nullptr)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(requires_grad)
    , is_volatile(is_volatile)
    , output_nr(0)
    , pyobj(nullptr)
{
  if (!this->data.defined()) {
    throw std::runtime_error("Variable data is NULL");
  }
}

Variable::Variable(
  at::Tensor data,
  std::shared_ptr<Function> grad_fn)
    : data(data)
    , grad_fn(grad_fn)
    , grad(nullptr)
    , version_counter(new VariableVersion())
    , requires_grad(grad_fn->is_executable)
    , is_volatile(false)
    , output_nr(grad_fn->num_inputs++)
    , pyobj(nullptr)
{
  if (!this->data.defined()) {
    throw std::runtime_error("Variable data is NULL");
  }
}

auto Variable::get_grad_accumulator() -> std::shared_ptr<Function> {
  if (grad_fn) {
    throw std::logic_error("get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad) return nullptr;

  std::lock_guard<std::mutex> lock(grad_accumulator_lock);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(shared_from_this());
  grad_accumulator = result;
  return result;
}

}} // namespace torch::autograd
