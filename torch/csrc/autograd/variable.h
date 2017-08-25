#pragma once

#include <Python.h>
#include <mutex>
#include <memory>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/VariableTensor.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct Variable : public at::Tensor {
  Variable() : Tensor() {}
  Variable(VariableTensor * self, bool retain) : Tensor(self, retain) {}
  Variable(const Variable & rhs) noexcept : Tensor(rhs) {}
  Variable(Variable && rhs) noexcept : Tensor(std::move(rhs)) {}
  /*implicit*/ Variable(Tensor const & rhs) : Tensor(rhs) {}
  /*implicit*/ Variable(Tensor && rhs) noexcept : Tensor(std::move(rhs)) {}

  Variable & operator=(Tensor && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Variable & operator=(Tensor const & rhs) & {
    //Tensor ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
    Variable(rhs).swap(*this);
    return *this;
  }
  Variable & operator=(const Variable & rhs) & {
    //Tensor ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
    Variable(rhs).swap(*this);
    return *this;
  }
  Variable & operator=(Tensor const & rhs) && {
    assign_(rhs);
    return *this;
  }

  VariableTensor* get() const { return static_cast<VariableTensor*>(pImpl); }

  const Tensor & data() const { return get()->data; }
        Tensor & data()       { return get()->data; }

  void set_data(const Tensor & data) { get()->data = data; }

  const Tensor & grad() const { return get()->grad; }
        Tensor & grad()       { return get()->grad; }

  const std::shared_ptr<Function>& grad_fn() const { return get()->grad_fn; };
        std::shared_ptr<Function>& grad_fn()       { return get()->grad_fn; };

  int current_version() const { return **get()->version_counter; }
  int output_nr() const { return get()->output_nr; }

  bool requires_grad() const { return get()->requires_grad; }
  void set_requires_grad(bool requires_grad) { get()->requires_grad = requires_grad; }

  bool is_volatile() const { return get()->is_volatile; }
  void set_volatile(bool is_volatile) { get()->is_volatile = is_volatile; }

  operator Tensor() const { return Tensor(pImpl, true); }
};

}} // namespace torch::autograd
