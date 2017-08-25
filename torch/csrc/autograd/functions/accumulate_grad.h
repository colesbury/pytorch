#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct AccumulateGrad : public Function {
  AccumulateGrad(at::Tensor variable);

  virtual variable_list apply(const variable_list& inputs) override;
  void acc_inplace(at::Tensor grad, at::Tensor new_grad);

  at::Tensor variable; // FIXME: weak
  at::Tensor variable_grad; // FIXME: weak
};

}}
