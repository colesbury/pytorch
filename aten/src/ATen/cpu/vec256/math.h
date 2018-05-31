#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#include "vec256_float.h"
#include "vec256_double.h"
#include "vec256_int.h"

namespace at { namespace vec256 {

template <typename T>
T fmadd(const T& a, const T& b, const T& c) {
  return a * b + c;
}

#ifdef __AVX2__
template <>
Vec256<float> fmadd(const Vec256<float>& a, const Vec256<float>& b, const Vec256<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}
template <>
Vec256<double> fmadd(const Vec256<double>& a, const Vec256<double>& b, const Vec256<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}
#endif

}}  // namespace at::vec256
