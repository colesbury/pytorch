#pragma once

#include <ATen/ATen.h>
#include <ATen/optional.h>
#include "CapabilityDispatch.h"

namespace at {
namespace native {

using var_fn = void(*)(Tensor&, const Tensor&, at::optional<int64_t>, bool unbiased);

extern DispatchStub<var_fn> var_kernel;

}
}
