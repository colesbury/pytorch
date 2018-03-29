#include "ATen/native/cpu/VarianceKernel.h"

#include <numeric>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at {
namespace native {

using namespace vec256;

static inline int64_t round_down(int64_t a, int64_t m) {
  return a - (a % m);
}

// Computes the harmonic sequence 1, 1/2, 1/3, ...
struct HarmonicSequence {
  using Vec = Vec256<double>;

  HarmonicSequence() {
    Vec::array_t init;
    for (int i = 0; i != Vec::size; i++) {
      init[i] = i + 1;
    }
    counts.load(init);
    (1.0 / counts).store(factors);
  }

  double next() {
    if (idx == Vec::size) {
      counts += 1;
      (1.0 / counts).store(factors);
      idx = 0;
    }
    return factors[idx++];
  }

  Vec counts = 0;
  Vec::array_t factors;
  int64_t idx = 0;
};

template <typename scalar_t>
static int64_t var128_v2(const scalar_t* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride);

template <>
int64_t var128_v2(const double* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<double>;
  Vec mean[4] = {0, 0, 0, 0};
  Vec M2[4] = {0, 0, 0, 0};

  HarmonicSequence harmonic;

  int count = 0;
  for (int64_t row = 0; row != rows; row++) {
    double factor = harmonic.next();
    for (int j = 0; j != 4; j++) {
      auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
      auto delta = val - mean[j];
      mean[j] = fmadd(delta, Vec(factor), mean[j]);
      auto delta2 = val - mean[j];
      M2[j] = fmadd(delta, delta2, M2[j]);
    }
    count++;
  }

  for (int j = 0; j != 4; j++) {
    mean[j].store(&mean_out[j * Vec::size]);
    M2[j].store(&m2_out[j * Vec::size]);
  }

  return count;
}

template <>
int64_t var128_v2(const float* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<double>;

  // double tmp1[4] = {1.0, 2.0, 3.0, 4.0};
  // double tmp2[4] = {10.0, 12.0, 13.0, 14.0};
  // Vec a = Vec::s_load(tmp1);
  // Vec b = Vec::s_load(tmp2);
  // auto tmp = cast_float(a, b);
  // std::cout << a << "\n";
  // std::cout << b << "\n";
  // std::cout << tmp << "\n";
  // a = cast_double(tmp.low());
  // b = cast_double(tmp.high());
  // tmp = cast_float(a, b);
  // std::cout << a << "\n";
  // std::cout << b << "\n";
  // std::cout << tmp << "\n";

  for (int offset = 0; offset != 32; offset += 16) {
    Vec mean[4] = {0, 0, 0, 0};
    Vec M2[4] = {0, 0, 0, 0};

    HarmonicSequence harmonic;

    auto update = [&](int j, double factor, Vec val) {
      auto delta = val - mean[j];
      mean[j] = fmadd(delta, Vec(factor), mean[j]);
      auto delta2 = val - mean[j];
      M2[j] = fmadd(delta, delta2, M2[j]);
    };

    for (int64_t row = 0; row != rows; row++) {
      double factor = harmonic.next();
      auto floats = Vec256<float>::s_load(&data[row * stride + offset]);
      update(0, factor, cast_double(floats.low()));
      update(1, factor, cast_double(floats.high()));
      floats.load(&data[row * stride + Vec256<float>::size + offset]);
      update(2, factor, cast_double(floats.low()));
      update(3, factor, cast_double(floats.high()));
    }

    for (int j = 0; j < 4; j++) {
      // auto meanf = cast_float(mean[j], mean[j + 1]);
      // auto M2f = cast_float(M2[j], M2[j + 1]);
      mean[j].store(&mean_out[j * Vec::size + offset]);
      M2[j].store(&m2_out[j * Vec::size + offset]);
    }
  }

  return rows;
}

template<typename scalar_t>
struct VarReduction {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);

  using Vec = Vec256<scalar_t>;

  static void apply(Tensor& res, const Tensor& self, at::optional<int64_t> dim) {
    internal::init_tbb_num_threads();

    auto out = res.data<scalar_t>();
    auto data = self.data<scalar_t>();
    auto numel = self.numel();
    if (!dim.has_value()) {
      *out = reduce_all(data, numel);
      return;
    }
  }

  static scalar_t reduce_all(const scalar_t* data, int size) {
    double mean[WIDTH];
    double M2[WIDTH];

    int64_t k = size / WIDTH;
    int64_t count = var128_v2(data, mean, M2, k, WIDTH);

    for (int step = 1; step != WIDTH; step = step << 1) {
      for (int j = 0; j < WIDTH; j += step * 2) {
        auto delta = mean[j] - mean[j + step];
        mean[j] = (mean[j] + mean[j + step]) / 2;
        M2[j] += M2[j + step] + delta * delta * count / 2;
      }
      count *= 2;
    }

    return M2[0] / count;
  }

  static int64_t var_twopass(const scalar_t* data, scalar_t* mean_out, scalar_t* m2_out, int64_t rows, int64_t stride) {
    Vec mean[4] = {0, 0, 0, 0};
    Vec c[4] = {0, 0, 0, 0};

    int count = 0;
    for (int64_t row = 0; row != rows; row++) {
      for (int j = 0; j != 4; j++) {
        auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
        auto y = val - c[j];
        auto t = mean[j] + y;
        c[j] = (t - mean[j]) - y;
        mean[j] = t;
      }
      count++;
    }

    for (int j = 0; j != 4; j++) {
      mean[j] /= count;
    }

    Vec M2[4] = {0, 0, 0, 0};
    for (int64_t row = 0; row != rows; row++) {
      for (int j = 0; j != 4; j++) {
        auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
        auto delta = (val - mean[j]);
        M2[j] += delta * delta;
      }
    }

    for (int j = 0; j != 4; j++) {
      mean[j].store(&mean_out[j * Vec::size]);
      M2[j].store(&m2_out[j * Vec::size]);
    }

    return count;
  }

  static int64_t var128(const scalar_t* data, scalar_t* mean_out, scalar_t* m2_out, int64_t rows, int64_t stride) {
    Vec mean[4] = {0, 0, 0, 0};
    Vec M2[4] = {0, 0, 0, 0};

    Vec counts;
    scalar_t init[Vec::size];
    scalar_t factors[Vec::size];
    for (int j = 0; j != Vec::size; j++) {
      init[j] = j + 1;
      factors[j] = 1.0 / init[j];
    }
    counts.load(init);
    uint32_t count_idx = 0;

    int count = 0;
    for (int64_t row = 0; row != rows; row++) {
      if (count_idx == Vec::size) {
        count_idx = 0;
        counts += Vec::size;
        auto factors_vec = Vec(1.0) / counts;
        factors_vec.store(factors);
      }
      scalar_t factor = factors[count_idx];
      for (int j = 0; j != 4; j++) {
        auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
        auto delta = val - mean[j];
        mean[j] = fmadd(delta, Vec(factor), mean[j]);
        auto delta2 = val - mean[j];
        M2[j] = fmadd(delta, delta2, M2[j]);
      }
      count++;
      count_idx++;
    }

    for (int j = 0; j != 4; j++) {
      mean[j].store(&mean_out[j * Vec::size]);
      M2[j].store(&m2_out[j * Vec::size]);
    }

    return count;
  }
};

static void var_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim, bool unbiased) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "var", [&] {
    VarReduction<scalar_t>::apply(result, self, dim);
  });
}

REGISTER_DISPATCH(var_kernel, &var_kernel_impl);

}}
