#include "Functions.h"

// ${generated_comment}

using namespace at;

namespace torch { namespace autograd {

Tensor norm_backward(const Tensor & grad, const Tensor & self, const Scalar & p_) {
  auto p = p_.toDouble();
  if (p == 2.0) {
    return self * (grad / self.norm(2));
  } else {
    throw std::runtime_error("norm_backward(): NYI");
  }
}

Tensor norm_backward(const Tensor & grad, const Tensor & self, const Scalar & p, int64_t dim, bool keepdim) {
  throw std::runtime_error("norm_backward(dim): NYI");
}

Tensor reduce_to(const Tensor & self, IntList sizes) {
  Tensor result = self;
  while (result.dim() > self.dim()) {
    result = result.sum(0);
  }
  for (int64_t i = 0; i < result.dim(); ++i) {
    if (sizes[i] == 1 && result.sizes()[i] > 1) {
      result = result.sum(i, true);
    }
  }
  return result;
}


${autograd_function_definitions}

}} // namespace torch::autograd
