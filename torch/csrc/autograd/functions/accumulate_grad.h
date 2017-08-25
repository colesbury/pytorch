#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct AccumulateGrad : public Function {
  AccumulateGrad(Variable variable);

  virtual variable_list apply(const variable_list& inputs) override;
  void acc_inplace(Variable grad, Variable new_grad);

  Variable variable; // FIXME: weak
  Variable variable_grad; // FIXME: weak
};

}}
