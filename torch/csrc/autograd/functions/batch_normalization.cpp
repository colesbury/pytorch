#include "batch_normalization.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/nn/THNN_generic.h"

namespace torch { namespace autograd {

using thpp::Tensor;

auto BatchNormForward::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() != 3) throw std::runtime_error("expected three inputs");

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];

  auto output = input->data->newTensor();
  output->resizeAs(*input->data);

  std::unique_ptr<Tensor> save_mean(output->newTensor());
  save_mean->resizeAs(*running_mean);
  std::unique_ptr<Tensor> save_std(output->newTensor());
  save_std->resizeAs(*running_var);

  torch::nn::BatchNormalization_updateOutput(
      input->data.get(),
      output.get(),
      weight ? weight->data.get() : nullptr,
      bias ? bias->data.get() : nullptr,
      running_mean.get(),
      running_var.get(),
      save_mean.get(),
      save_std.get(),
      training,
      momentum,
      eps);

  auto creator = std::make_shared<BatchNormBackward>(
      flags(inputs),
      std::unique_ptr<thpp::Tensor>(running_mean->clone_shallow()),
      std::unique_ptr<thpp::Tensor>(running_var->clone_shallow()),
      std::move(save_mean),
      std::move(save_std),
      input->save(),
      save_optional(weight.get()),
      save_optional(bias.get()),
      training,
      momentum,
      eps);
  variable_list results(1);
  results[0] = std::make_shared<THVariable>(std::move(output), creator);
  return results;
};

auto BatchNormBackward::apply(const variable_list& gradOutputs) -> variable_list {
  auto& input = this->input.unpack();
  auto& weight = this->weight.unpack();
  auto& bias = this->bias.unpack();

  std::unique_ptr<Tensor> gradInput = input->newTensor();
  gradInput->resizeAs(*input);

  std::unique_ptr<Tensor> gradWeight;
  if (weight) {
    gradWeight = weight->newTensor();
    gradWeight->resizeAs(*weight);
    gradWeight->zero();
  }

  std::unique_ptr<Tensor> gradBias;
  if (bias) {
    gradBias = bias->newTensor();
    gradBias->resizeAs(*bias);
    gradBias->zero();
  }

  torch::nn::BatchNormalization_backward(
      input.get(),
      gradOutputs[0]->data.get(),
      gradInput.get(),
      gradWeight.get(),
      gradBias.get(),
      weight.get(),
      running_mean.get(),
      running_var.get(),
      save_mean.get(),
      save_std.get(),
      training,
      1.0,
      eps);

  variable_list results(3);
  results[0] = THVariable::of(std::move(gradInput));
  results[1] = THVariable::of(std::move(gradWeight));
  results[2] = THVariable::of(std::move(gradBias));
  return results;
};

}} // namespace torch::autograd
