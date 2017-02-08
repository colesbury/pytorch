#include "Handles.h"

#include <unordered_map>
#include <mutex>
#include <THC/THC.h>
#include "Exceptions.h"

extern THCState* state;

namespace torch { namespace cudnn {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle() : handle(NULL) {
    CHECK(cudnnCreate(&handle));
  }
  ~Handle() {
    if (handle) {
      cudnnDestroy(handle);
    }
  }
};

std::mutex mutex;
std::unordered_map<int, Handle> handles;

}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  // TODO: NOT THREAD SAFE
  std::lock_guard<std::mutex> guard(mutex);
  auto handle = handles[device].handle;
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  return handle;
}

}} // namespace torch::cudnn
