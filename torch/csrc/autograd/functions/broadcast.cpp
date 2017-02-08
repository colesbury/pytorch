#include "torch/csrc/autograd/functions/broadcast.h"

#include "torch/csrc/nccl/reductions.h"
#include "torch/csrc/utils/auto_gpu.h"


namespace torch { namespace autograd {

namespace {

const size_t kMiB = 1048576;

auto flatten(const std::vector<variable_list>& inputs) -> variable_list {
  variable_list output;
  for (auto& input : inputs) {
    for (auto& var : input) {
      output.push_back(var);
    }
  }
  return output;
}

}

auto Broadcast::broadcast(const variable_list& inputs, std::vector<variable_list>& results) -> void {
  tensor_list tensors;
  tensors.reserve(inputs.size());
  for (auto& input : inputs) {
    tensors.emplace_back(input->data->clone_shallow());
  }

  auto& comms = torch::nccl::communicator(devices);
  auto output_tensors = torch::nccl::broadcast(comms, tensors, devices);

  auto creator = std::make_shared<Reduce>(tensors[0]->getDevice(), flags(inputs));
  for (size_t i = 0; i < devices.size(); ++i) {
    auto stream = torch::nccl::get_stream(devices[i]);
    auto event = std::make_shared<CudaVariableEvent>(stream);
    for (auto& tensor : output_tensors[i]) {
      auto var = std::make_shared<Variable>(std::move(tensor), creator);
      var->event = event;
      results[i].push_back(std::move(var));
    }
  }
}

auto Broadcast::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() == 0) throw std::runtime_error("expected at least one input");

  torch::nccl::synchronize_before(devices);

  std::vector<variable_list> results(devices.size());

  variable_list buffer;
  uint64_t size = 0;
  size_t limit = kMiB;
  for (auto& input : inputs) {
    auto& tensor = input->data;
    size_t tensor_bytes = tensor->numel() * tensor->elementSize();
    if (size > 0 && size + tensor_bytes > limit) {
      broadcast(buffer, results);
      buffer.clear();
      size = 0;
      limit = 10 * kMiB;
    }
    buffer.push_back(input);
    size += tensor_bytes;
  }
  if (!buffer.empty()) {
    broadcast(buffer, results);
  }

  return flatten(results);
}

auto Reduce::apply(const variable_list& inputs) -> variable_list {
  if (inputs.size() == 0) throw std::runtime_error("expected at least one input");

  int size = previous_functions.size();
  AutoGPU guard(device);

  variable_list results(size);
  for (int i = 0; i < size; ++i) {
    if (needs_grad[i]) {
      auto data = inputs[i]->data->newTensor();
      data->resizeAs(*inputs[i]->data);
      results[i] = std::make_shared<Variable>(std::move(data), false, true);
    }
  }
  return results;
}

}} // namespace torch::autograd
