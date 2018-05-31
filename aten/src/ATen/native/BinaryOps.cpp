#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/native/TensorIterator.h"
#include "cpu/BinaryOpsKernel.h"
#include <time.h>

namespace at {
namespace native {


  struct timestamp {
    timestamp() {
      clock_gettime(CLOCK_MONOTONIC, &tv);
    }
    int64_t elapsed_time() {
      return elapsed_time(timestamp());
    }
    int64_t elapsed_time(const timestamp& other) {
      int64_t ds = (other.tv.tv_sec - tv.tv_sec);
      ds *= 1000000000;
      ds += (other.tv.tv_nsec - tv.tv_nsec);
      return ds;
    }
    struct timespec tv;
  };

DispatchStub<binary_fn> add_stub;

Tensor add2(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(self, other);
  add_stub.call(iter.backend(), iter, alpha);
  return iter.output();
}

Tensor& add2_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(self, other, result);
  add_stub(iter, alpha);
  return result;
}

Tensor& add2_(Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(self, other, self);
  add_stub(iter, alpha);
  return self;
}

}
}  // namespace at
