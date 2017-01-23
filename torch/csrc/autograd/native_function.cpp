#include <Python.h>
#include "THP.h"

#include <THPP/THPP.h>

using thpp::Tensor;

namespace torch { namespace autograd {

class BatchNorm : public NativeFunction {
  BatchNorm(std::unique_ptr<Tensor> running_mean,
            std::unique_ptr<Tensor> running_var,
            bool training,
            double momentum,
            double eps)
    : running_mean(std::move(running_mean))
    , running_var(std::move(running_var))
    , training(training)
    , momentum(momentum)
    , eps(eps) {}

  virtual variable_list forward(const variable_list& inputs) override;
  virtual tensor_list backward(const tensor_list& gradOutputs, bool retain_variables) override;

  std::unique_ptr<Tensor> running_mean;
  std::unique_ptr<Tensor> running_var;
  bool training;
  double momentum;
  double eps;
};

auto BatchNorm::forward(const variable_list& inputs) -> variable_list {
  if (inputs.size() != 3) throw std::runtime_error("expected three inputs");

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];

  int num_features = input->data->sizes()[1];
  auto output = input->data->newTensor();
  output->resizeAs(*input->data);

  auto var = std::make_shared<THVariable>(std::move(output), 0, 0);
  // TODO: var->creator
  return variable_list({ std::move(var) });
};

auto BatchNorm::backward(const tensor_list& gradOutputs, bool retain_variables) -> tensor_list {
  return tensor_list();
};

}} // namespace torch::autograd
