#pragma once

#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/native_function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct BatchNormForward : public NativeFunction {
  BatchNormForward(
      std::unique_ptr<thpp::Tensor> running_mean,
      std::unique_ptr<thpp::Tensor> running_var,
      bool training,
      double momentum,
      double eps)
    : running_mean(std::move(running_mean))
    , running_var(std::move(running_var))
    , training(training)
    , momentum(momentum)
    , eps(eps) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::unique_ptr<thpp::Tensor> running_mean;
  std::unique_ptr<thpp::Tensor> running_var;
  bool training;
  double momentum;
  double eps;
};

struct BatchNormBackward : public NativeFunction {
  BatchNormBackward(
      FunctionFlags flags,
      std::unique_ptr<thpp::Tensor> running_mean,
      std::unique_ptr<thpp::Tensor> running_var,
      std::unique_ptr<thpp::Tensor> save_mean,
      std::unique_ptr<thpp::Tensor> save_std,
      SavedVariable input,
      SavedVariable weight,
      SavedVariable bias,
      bool training,
      double momentum,
      double eps)
    : NativeFunction(std::move(flags))
    , running_mean(std::move(running_mean))
    , running_var(std::move(running_var))
    , save_mean(std::move(save_mean))
    , save_std(std::move(save_std))
    , input(std::move(input))
    , weight(std::move(weight))
    , bias(std::move(bias))
    , training(training)
    , momentum(momentum)
    , eps(eps) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;

  std::unique_ptr<thpp::Tensor> running_mean;
  std::unique_ptr<thpp::Tensor> running_var;
  std::unique_ptr<thpp::Tensor> save_mean;
  std::unique_ptr<thpp::Tensor> save_std;
  SavedVariable input;
  SavedVariable weight;
  SavedVariable bias;
  bool training;
  double momentum;
  double eps;
};

}}
