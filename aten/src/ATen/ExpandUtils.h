#pragma once

#include "ATen/Tensor.h"
#include <sstream>
#include <tuple>

namespace at {

std::vector<int64_t> infer_size(IntList a, IntList b);
std::tuple<std::vector<int64_t>, std::vector<int64_t> > inferExpandGeometry(const Tensor &tensor, IntList sizes);

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(tensor.sizes()));
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  return std::make_tuple(to_expand1.expand(tensor.sizes()), to_expand2.expand(tensor.sizes()));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size));
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = infer_size(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size(expanded_size12, to_expand3.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size), to_expand3.expand(expanded_size));
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(sizes));
}

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sizes();
      first = false;
    } else {
      sizes = infer_size(sizes, to_expand[i].sizes());
    }
  }

  std::vector<Tensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = to_expand[i].expand(sizes);
    }
  }
  return result;
}

}
