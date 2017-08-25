#pragma once

#include <ATen/ATen.h>
#include <memory>
#include <unordered_map>

namespace torch { namespace autograd {

struct VariableTypeRegistry {

  static at::Type* get(const at::Tensor& tensor);
  static at::Type* get(const at::Type& baseType);

private:
  static std::unordered_map<const at::Type*, std::unique_ptr<at::Type>> types;
};

}} // namespace torch::autograd
