#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#include "vec256_float.h"
#include "vec256_double.h"

namespace at {
namespace vec256 {

#ifdef __AVX__
Vec256<double> inline cast_double(__m128 value) {
  return _mm256_cvtps_pd(value);
}
#else
Vec256<double> inline cast_double(Vec128<float> value) {
  // return _mm256_cvtps_pd(value);
  return Vec256<double>();
}
#endif

Vec256<double> inline cast_double(Vec256<float> value, int part) {
#ifdef __AVX__
  return _mm256_cvtps_pd(value.extract(part));
#else
  return Vec256<double>();
#endif
}

Vec256<float> inline cast_float(Vec256<double> low, Vec256<double> high) {
#ifdef __AVX__
  auto a = _mm256_cvtpd_ps(low);
  auto b = _mm256_cvtpd_ps(high);
  return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
#else
  return Vec256<float>();
#endif
}

}}
