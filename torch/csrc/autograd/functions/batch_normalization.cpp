#include "batch_normalization.h"

#include <THC/THC.h>
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/nn/THNN_generic.h"
#ifdef WITH_CUDNN
#include "torch/csrc/cudnn/BatchNorm.h"
#include "torch/csrc/cudnn/Handles.h"
#include "torch/csrc/cudnn/Types.h"
#endif

extern THCState* state;

namespace torch { namespace autograd {

using thpp::Tensor;

auto BatchNormForward::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() != 3) throw std::runtime_error("expected three inputs");

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->data->isCuda()
               && input->data->type() != thpp::Type::HALF
               && weight && bias);
#endif

  auto output = input->data->newTensor();
  output->resizeAs(*input->data);

  std::unique_ptr<Tensor> save_mean(output->newTensor());
  save_mean->resizeAs(*running_mean);
  std::unique_ptr<Tensor> save_std(output->newTensor());
  save_std->resizeAs(*running_var);

  if (use_cudnn) {
#ifdef WITH_CUDNN
    torch::cudnn::cudnn_batch_norm_forward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(*input->data),
        (THVoidTensor*)input->data->cdata(),
        (THVoidTensor*)output->cdata(),
        (THVoidTensor*)weight->data->cdata(),
        (THVoidTensor*)bias->data->cdata(),
        (THVoidTensor*)running_mean->cdata(),
        (THVoidTensor*)running_var->cdata(),
        (THVoidTensor*)save_mean->cdata(),
        (THVoidTensor*)save_std->cdata(),
        training,
        momentum,
        eps);
#endif
  } else {
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
  }

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

auto BatchNormBackward::apply(const variable_list& grad_outputs) -> variable_list {
  auto& input = this->input.unpack();
  auto& weight = this->weight.unpack();
  auto& bias = this->bias.unpack();

  bool use_cudnn = false;
#ifdef WITH_CUDNN
  use_cudnn = (input->isCuda()
               && input->type() != thpp::Type::HALF
               && weight && bias && training);
#endif

  std::unique_ptr<Tensor> grad_input = input->newTensor();
  grad_input->resizeAs(*input);

  std::unique_ptr<Tensor> grad_weight;
  if (weight) {
    grad_weight = weight->newTensor();
    grad_weight->resizeAs(*weight);
    if (!use_cudnn) {
      grad_weight->zero();
    }
  }

  std::unique_ptr<Tensor> grad_bias;
  if (bias) {
    grad_bias = bias->newTensor();
    grad_bias->resizeAs(*bias);
    grad_bias->zero();
    if (!use_cudnn) {
      grad_bias->zero();
    }
  }

  if (use_cudnn) {
    torch::cudnn::cudnn_batch_norm_backward(
        state,
        torch::cudnn::getCudnnHandle(),
        torch::cudnn::getCudnnDataType(*input),
        (THVoidTensor*)input->cdata(),
        (THVoidTensor*)grad_outputs[0]->data->cdata(),
        (THVoidTensor*)grad_input->cdata(),
        (THVoidTensor*)grad_weight->cdata(),
        (THVoidTensor*)grad_bias->cdata(),
        (THVoidTensor*)weight->cdata(),
        (THVoidTensor*)running_mean->cdata(),
        (THVoidTensor*)running_var->cdata(),
        (THVoidTensor*)save_mean->cdata(),
        (THVoidTensor*)save_std->cdata(),
        training,
        eps);
  } else {
    torch::nn::BatchNormalization_backward(
        input.get(),
        grad_outputs[0]->data.get(),
        grad_input.get(),
        grad_weight.get(),
        grad_bias.get(),
        weight.get(),
        running_mean.get(),
        running_var.get(),
        save_mean.get(),
        save_std.get(),
        training,
        1.0,
        eps);
  }

  variable_list results(3);
  results[0] = THVariable::of(std::move(grad_input));
  results[1] = THVariable::of(std::move(grad_weight));
  results[2] = THVariable::of(std::move(grad_bias));
  return results;
};

}} // namespace torch::autograd
