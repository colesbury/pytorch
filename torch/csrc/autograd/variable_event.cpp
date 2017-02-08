#include "torch/csrc/autograd/variable_event.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_gpu.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <THC/THC.h>
extern THCState* state;
#endif

namespace torch { namespace autograd {

#ifdef WITH_CUDA
CudaVariableEvent::CudaVariableEvent(THCStream* stream) : event(nullptr) {
  AutoGPU guard(stream->device);
  THCudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  THCudaCheck(cudaEventRecord(event, stream->stream));
}

CudaVariableEvent::~CudaVariableEvent() {
  if (event) {
    cudaEventDestroy(event);
  }
}

auto CudaVariableEvent::await(Variable& var) -> void {
  if (event) {
    AutoGPU guard(var.data->getDevice());
    cudaStream_t stream = THCState_getCurrentStream(state);
    // THCudaCheck(cudaStreamWaitEvent(stream, event, 0));
    // TODO: record stream in caching allocator
  }
}

#endif

}} // namespace torch::autograd
