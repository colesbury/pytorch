#pragma once

#include "ATen/Tensor.h"
#include "ATen/ScalarType.h"

namespace at { namespace indexing {

Tensor indexTake(const Tensor & self, TensorList indices);

Tensor & indexPut(Tensor & self, TensorList indices, const Tensor & value);

}} // namespace at::indexing
