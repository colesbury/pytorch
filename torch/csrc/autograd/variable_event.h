#pragma once

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <THC/THCStream.h>
#endif

namespace torch { namespace autograd {

struct Variable;

struct VariableEvent {
  virtual void await(Variable& var) = 0;
};

#ifdef WITH_CUDA
struct CudaVariableEvent : public VariableEvent {
  explicit CudaVariableEvent(THCStream* stream);
  ~CudaVariableEvent();

  virtual void await(Variable& var) override;

  cudaEvent_t event;
};
#endif

}} // namespace torch::autograd
