#include "torch/csrc/nccl/reductions.h"

#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/auto_stream.h"

#include <mutex>
#include <stdint.h>
#include <THC/THC.h>
#include <unordered_map>

extern THCState* state;

namespace torch { namespace nccl {

std::unordered_map<std::string, comm_list_t> communicators;
std::vector<THCStream*> streams;
std::vector<THCStream*> streams2;
std::vector<cudaEvent_t> events;

std::vector<THCStream*>& get_streams() {
  static std::once_flag once;
  call_once(once, []{
    int num_devices = THCState_getNumDevices(state);
    streams.resize(num_devices);
    streams2.resize(num_devices);
    events.resize(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      AutoGPU guard(i);
      streams[i] = THCStream_newWithPriority(cudaStreamNonBlocking, -1);
      streams2[i] = THCStream_newWithPriority(cudaStreamNonBlocking, -1);
      THCudaCheck(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }
  });
  return streams;
}

THCStream* get_stream(int device) {
  auto& streams = get_streams();
  THAssert(device >= 0 && (size_t)device < streams.size());
  return streams[device];
}

void CHECK_NCCL(ncclResult_t res) {
  if (res != ncclSuccess) {
    throw std::runtime_error("NCCL error");
  }
}

ncclDataType_t get_datatype(thpp::Type type) {
  switch (type) {
    case thpp::Type::CHAR: return ncclChar;
    case thpp::Type::INT: return ncclInt;
    case thpp::Type::HALF: return ncclHalf;
    case thpp::Type::FLOAT: return ncclFloat;
    case thpp::Type::DOUBLE: return ncclDouble;
    case thpp::Type::LONG: return ncclInt64;
    case thpp::Type::ULONG: return ncclUint64;
    default:
      throw std::runtime_error("No NCCL equivalent for data type: " +
        std::to_string(static_cast<char>(type)));
  }
}

std::string get_key(const std::vector<int>& devices) {
  std::string key;
  for (int device : devices) {
    key += std::to_string(device) + ",";
  }
  return key;
}

comm_list_t& communicator(const std::vector<int>& devices) {
  std::string key = get_key(devices);
  auto it = communicators.find(key);
  if (it != communicators.end()) {
    return it->second;
  }
  comm_list_t comms(devices.size());
  CHECK_NCCL(ncclCommInitAll(comms.data(), devices.size(), devices.data()));
  return (communicators[key] = std::move(comms));
}

std::unique_ptr<thpp::Tensor> flatten(const tensor_list& inputs) {
  if (inputs.size() == 0) {
    return std::unique_ptr<thpp::Tensor>();
  }

  int device = inputs[0]->getDevice();
  AutoGPU guard(device);
  AutoStream stream(get_stream(device));

  if (inputs.size() == 1) {
    auto contiguous = inputs[0]->contiguous();
    auto flat = inputs[0]->newTensor();
    flat->set(*contiguous);
    flat->resize({ flat->numel() });
    return flat;
  }

  long long numel = 0;
  for (auto& tensor : inputs) {
    numel += tensor->numel();
  }

  auto flat = inputs[0]->newTensor();
  flat->resize({ numel });
  long long offset = 0;
  for (auto& tensor : inputs) {
    auto slice = flat->newTensor();
    slice->narrow(*flat, 0, offset, tensor->numel());
    // printf("COPY\n");
    slice->copy(*tensor);
    offset += tensor->numel();
  }
  return flat;
}

auto synchronize_before(const std::vector<int>& devices) -> void {
  auto& streams = get_streams();
  // cudaEvent_t event;
  // {
  //   AutoGPU guard(devices[0]);
  //   THCudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  //   THCudaCheck(cudaEventRecord(event, THCState_getCurrentStream(state)));
  // }
  // for (int device : devices) {
  //   AutoGPU guard(device);
  //   THCudaCheck(cudaStreamWaitEvent(streams[device]->stream, event, 0));
  //   break;
  // }
  // THCudaCheck(cudaEventDestroy(event));
}

auto broadcast(
    const comm_list_t& comms,
    const tensor_list& inputs,
    const std::vector<int>& devices) -> std::vector<tensor_list> {

  std::vector<std::unique_ptr<thpp::Tensor>> buffers(devices.size());
  buffers[0] = flatten(inputs);
  for (size_t i = 1; i < devices.size(); i++) {
    AutoGPU guard(devices[i]);
    AutoStream stream(streams[devices[i]]);
    buffers[i] = buffers[0]->newTensor();
    buffers[i]->resizeAs(*buffers[0]);
  }

  ncclDataType_t datatype = get_datatype(buffers[0]->type());

  // TODO: lock CUDA free mutex
  for (size_t i = 0; i < devices.size(); i++) {
    AutoGPU guard(devices[i]);
    int size = buffers[i]->numel();
    CHECK_NCCL(ncclBcast(
        buffers[i]->data(), size, datatype, devices[0], comms[i],
        NULL));//streams[devices[i]]->stream));
  }

  std::vector<tensor_list> outputs(devices.size());
  for (size_t i = 0; i < devices.size(); i++) {
    AutoGPU guard(devices[i]);
    auto& buffer = buffers[i];
    auto& list = outputs[i];
    list.reserve(inputs.size());

    long long offset = 0;
    for (auto& tensor : inputs) {
      auto slice = buffer->newTensor();
      slice->narrow(*buffer, 0, offset, tensor->numel());
      slice->resizeAs(*tensor);
      list.push_back(std::move(slice));
      offset += tensor->numel();
    }
  }
  return outputs;
}

}}  // namespace torch::nccl
