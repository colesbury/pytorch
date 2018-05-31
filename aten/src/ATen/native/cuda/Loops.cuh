#include <ATen/ATen.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/CUDATypeConversion.cuh>
#include <THC/THCNumerics.cuh>


// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

namespace at { namespace native {

template<int nt, int vt, typename func_t>
__launch_bounds__(nt, 4)
__global__ void generic_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int cta = blockIdx.x;
  int nv = nt * vt;
  int idx = nv * cta + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int N>
static OffsetCalculator<N> make_offset_calculator(const TensorIterator& iter) {
  AT_ASSERT(N == iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides);
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, func_t f) {
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = globalContext().getCurrentCUDAStream();
  generic_kernel<nt, vt><<<grid, block, 0, stream>>>(N, f);
}

template <typename T>
__device__ static void store_convert(void* out, T value, ScalarType dtype) {
  switch (dtype) {
    case at::kByte: *(uint8_t*)out = (uint8_t)value; break;
    case at::kChar: *(int8_t*)out = (int8_t)value; break;
    case at::kShort: *(int16_t*)out = (int16_t)value; break;
    case at::kInt: *(int32_t*)out = (int32_t)value; break;
    case at::kLong: *(int64_t*)out = (int64_t)value; break;
    case at::kHalf: *(half*)out = scalar_cast<half>(value); break;
    case at::kFloat: *(float*)out = (float)value; break;
    case at::kDouble: *(double*)out = (double)value; break;
  }
}

template <typename T>
__device__ static T load_convert(const void* in, ScalarType dtype) {
  switch (dtype) {
    case at::kByte: return scalar_cast<T>(*(uint8_t*)in);
    case at::kChar: return scalar_cast<T>(*(int8_t*)in);
    case at::kShort: return scalar_cast<T>(*(int16_t*)in);
    case at::kInt: return scalar_cast<T>(*(int32_t*)in);
    case at::kLong: return scalar_cast<T>(*(int64_t*)in);
    case at::kHalf: return scalar_cast<T>(*(half*)in);
    case at::kFloat: return scalar_cast<T>(*(float*)in);
    case at::kDouble: return scalar_cast<T>(*(double*)in);
  }
  assert(0);
  return T(0);
}

template<typename func_t>
inline void unary_kernel(TensorIterator& iter, func_t f) {
  char* out_data = (char*)iter.data_ptr(0);
  const char* in1_data = (char*)iter.data_ptr(1);

  using traits = unary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;

  int64_t numel = iter.numel();
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    int stride1 = strides[1];
    launch_kernel<512, 1>(numel, [=]__device__(int idx) {
      arg0_t* out = (arg0_t*)&out_data[stride0 * idx];
      arg1_t* in1 = (arg1_t*)&in1_data[stride1 * idx];
      *out = f(*in1);
    });
  } else if (!iter.needs_cast()) {
    auto offset_calc = make_offset_calculator<3>(iter);
    launch_kernel<128, 4>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)&out_data[offsets[0]];
      arg1_t* in1 = (arg1_t*)&in1_data[offsets[1]];
      *out = f(*in1);
    });
  } else {
    auto offset_calc = make_offset_calculator<3>(iter);
    ScalarType out_type = iter.dtype(0);
    ScalarType in1_type = iter.dtype(1);
    launch_kernel<128, 4>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg1_t in1 = load_convert<arg1_t>(&in1_data[offsets[1]], in1_type);
      arg0_t out = f(in1);
      store_convert(&out_data[offsets[0]], out, out_type);
    });
  }
}

template<typename func_t>
inline void binary_kernel(TensorIterator& iter, func_t f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      binary_kernel(sub_iter, f);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in1_data = (char*)iter.data_ptr(1);
  const char* in2_data = (char*)iter.data_ptr(2);

  using traits = binary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;
  using arg2_t = typename traits::arg2_t;

  int numel = iter.numel();

  if (iter.is_cpu_scalar(1)) {
    auto a = iter.scalar_value<arg1_t>(1);
    iter.remove_operand(1);
    unary_kernel(iter, [=]GPU_LAMBDA(arg2_t b) {
      return f(a, b);
    });
  } else if (iter.is_cpu_scalar(2)) {
    auto b = iter.scalar_value<arg2_t>(2);
    iter.remove_operand(2);
    unary_kernel(iter, [=]GPU_LAMBDA(arg1_t a) {
      return f(a, b);
    });
  } else if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    int stride1 = strides[1];
    int stride2 = strides[2];
    launch_kernel<512, 1>(numel, [=]__device__(int idx) {
      arg0_t* out = (arg0_t*)&out_data[stride0 * idx];
      arg1_t* in1 = (arg1_t*)&in1_data[stride1 * idx];
      arg2_t* in2 = (arg2_t*)&in2_data[stride2 * idx];
      *out = f(*in1, *in2);
    });
  } else if (!iter.needs_cast()) {
    auto offset_calc = make_offset_calculator<3>(iter);
    launch_kernel<128, 4>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)&out_data[offsets[0]];
      arg1_t* in1 = (arg1_t*)&in1_data[offsets[1]];
      arg2_t* in2 = (arg2_t*)&in2_data[offsets[2]];
      *out = f(*in1, *in2);
    });
  } else {
    auto offset_calc = make_offset_calculator<3>(iter);
    ScalarType out_type = iter.dtype(0);
    ScalarType in1_type = iter.dtype(1);
    ScalarType in2_type = iter.dtype(2);
    launch_kernel<128, 4>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg1_t in1 = load_convert<arg1_t>(&in1_data[offsets[1]], in1_type);
      arg2_t in2 = load_convert<arg2_t>(&in2_data[offsets[2]], in2_type);
      arg0_t out = f(in1, in2);
      store_convert(&out_data[offsets[0]], out, out_type);
    });
  }
}

}} // namespace at::native
