#pragma once

#include <memory>
#include <vector>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Broadcast : public Function {
  Broadcast(const std::vector<int>& devices) : devices(devices) {}
  Broadcast(const std::vector<int>& devices, FunctionFlags&& flags)
    : Function(std::move(flags))
    , devices(devices) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<int> devices;

private:
  void broadcast(const variable_list& inputs, std::vector<variable_list>& results);
};

struct Reduce : public Function {
  Reduce(int device) : device(device) {}
  Reduce(int device, FunctionFlags&& flags)
    : Function(std::move(flags))
    , device(device) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int device;
};

}} // namespace torch::autograd
