#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

static void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    binary_kernel(iter, [alpha]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a + alpha * b;
    });
  });
}

using binary_fn = void(*)(TensorIterator&, Scalar alpha);
extern DispatchStub<binary_fn> add_stub;
REGISTER_DISPATCH(add_stub, &add_kernel_cuda);

}} // namespace at::native
