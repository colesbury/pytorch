#pragma once

#include <array>
#include <cstdint>
#include <THC/THCIntegerDivider.cuh>

template<int NARGS>
struct OffsetCalculator {
  static const int MAX_DIMS = 25;

  struct offsets_t {
    __device__ uint32_t& operator[](int idx) {
      return values[idx];
    }
    uint32_t values[NARGS];
  };

  OffsetCalculator(int Dims, const int64_t* sizes, std::array<const int64_t*, NARGS> instrides) : Dims(Dims) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < Dims) {
        sizes_[i] = IntDivider<uint32_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<uint32_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        strides[i][arg] =  i < Dims ? instrides[arg][i] : 0;
      }
    }
  }

  __host__ __device__ offsets_t get(uint32_t linear_idx) const {
    offsets_t offsets;
    #pragma unroll
    for (int j = 0; j < NARGS; j++) {
      offsets[j] = 0;
    }

    #pragma unroll
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i == Dims) {
        break;
      }
      auto divmod = sizes_[i].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int j = 0; j < NARGS; j++) {
        offsets[j] += divmod.mod * strides[i][j];
      }

    }
    return offsets;
  }

  int Dims;
  IntDivider<uint32_t> sizes_[MAX_DIMS];
  uint32_t strides[MAX_DIMS][NARGS];
};
