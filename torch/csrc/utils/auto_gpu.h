#pragma once

// RAII structs to set CUDA device

#include <string>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

struct AutoGPU {
  explicit AutoGPU(int device=-1) {
    setDevice(device);
  }

  ~AutoGPU() {
#ifdef WITH_CUDA
    if (original_device != -1) {
      cudaSetDevice(original_device);
    }
#endif
  }

  inline bool setDevice(int device) {
#ifdef WITH_CUDA
    if (device == -1) {
      return false;
    }
    if (original_device == -1) {
      cudaCheck(cudaGetDevice(&original_device));
      if (device != original_device) {
        cudaCheck(cudaSetDevice(device));
      }
    } else {
      cudaCheck(cudaSetDevice(device));
    }
    return true;
#else
    return false;
#endif
  }

  int original_device = -1;

private:
#ifdef WITH_CUDA
  static void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
      std::string msg = "CUDA error (";
      msg += std::to_string(err);
      msg += "): ";
      msg += cudaGetErrorString(err);
      throw std::runtime_error(msg);
    }
  }
#endif
};
