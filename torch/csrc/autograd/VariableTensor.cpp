#include "torch/csrc/autograd/VariableTensor.h"

#include "torch/csrc/autograd/variable_type_registry.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"

using namespace at;

namespace torch { namespace autograd {

VariableTensor::VariableTensor(Tensor data)
  : TensorImpl(VariableTypeRegistry::get(data))
  , data(data)
  , grad()
  , version_counter(new VariableVersion())
  , requires_grad(false)
  , is_volatile(false)
  , output_nr(0)
  , pyobj(nullptr) {
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

VariableTensor::VariableTensor(Tensor data, std::shared_ptr<Function> grad_fn)
  : VariableTensor(data)
{
  this->grad_fn = grad_fn;
  requires_grad = grad_fn->is_executable;
  output_nr = grad_fn->num_inputs++;
}

VariableTensor::~VariableTensor() {
}

const char * VariableTensor::toString() const {
  return "Variable";
}

IntList VariableTensor::sizes() {
  return data.sizes();
}

int64_t VariableTensor::dim() {
  return data.dim();
}

const char * VariableTensor::typeString() {
  return "VariableType";
}

void * VariableTensor::unsafeGetTH(bool retain) {
  return data.unsafeGetTH(retain);
}

IntList VariableTensor::strides() {
  return data.strides();
}

Scalar VariableTensor::localScalar() {
  return data.pImpl->localScalar();
}

void VariableTensor::assign_(Scalar s) {
  data.assign_(s);
}

std::shared_ptr<Function> VariableTensor::get_grad_accumulator() {
  if (grad_fn) {
    throw std::logic_error("get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad) return nullptr;

  std::lock_guard<std::mutex> lock(grad_accumulator_lock);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(Tensor(this, true));
  grad_accumulator = result;
  return result;
}

}}
