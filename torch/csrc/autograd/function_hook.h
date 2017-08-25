#pragma once

#include <memory>
#include <vector>

// A hook that's called on gradients

namespace at { struct Tensor; }

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

struct FunctionPreHook {
  virtual variable_list operator()(const variable_list& grads) = 0;
};

struct FunctionPostHook {
  virtual variable_list operator()(const variable_list& grad_input, const variable_list& grad_output) = 0;
};

}} // namespace torch::autograd
