#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using binary_fn = void(*)(TensorIterator&, Scalar alpha);

extern DispatchStub<binary_fn> add_stub;

}} // namespace at::native
