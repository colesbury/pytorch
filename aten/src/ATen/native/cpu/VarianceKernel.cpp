#include "ATen/native/cpu/VarianceKernel.h"

#include <algorithm>
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
template<typename T>
struct HarmonicSequence {
  using Vec = Vec256<T>;

  HarmonicSequence() {
    T init[Vec::size];
    for (int i = 0; i != Vec::size; i++) {
      init[i] = i + 1;
    }
    counts.load(init);
    (1.0 / counts).store(factors);
  }

  T next() {
    if (idx == Vec::size) {
      counts += Vec::size;
      (1.0 / counts).store(factors);
      idx = 0;
    }
    return factors[idx++];
  }

  Vec counts;
  T factors[Vec::size];
  int64_t idx = 0;
};

template <typename scalar_t>
static int64_t var128_v2(const scalar_t* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride);

template <>
int64_t var128_v2(const double* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<double>;
  Vec mean[4] = {0, 0, 0, 0};
  Vec M2[4] = {0, 0, 0, 0};

  HarmonicSequence<double> harmonic;

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

static int64_t var128_v3(const float* data, float* mean_out, float* m2_out, float* c_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<float>;
  Vec c[4] = {0, 0, 0, 0};
  Vec mean[4] = {0, 0, 0, 0};

  // First estimate the mean using Kahan summation
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 4; j++) {
      auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
      auto y = val - c[j];
      auto t = mean[j] + y;
      c[j] = (t - mean[j]) - y;
      mean[j] = t;
    }
  }

  float inv_rows = 1.0f / rows;
  for (int j = 0; j != 4; j++) {
    mean[j] *= inv_rows;
    c[j] *= inv_rows;
    c[j].store(&c_out[j * Vec::size]);
  }

  Vec M2[4] = {0, 0, 0, 0};
  Vec correction[4] = {0, 0, 0, 0};
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 4; j++) {
      auto val = Vec::s_load(&data[row * stride + j * Vec::size]);
      auto delta = val - mean[j];
      M2[j] = fmadd(delta, delta, M2[j]);
      correction[j] += delta;
    }
  }

  for (int j = 0; j != 4; j++) {
    M2[j] -= inv_rows * correction[j] * correction[j];
  }

  for (int j = 0; j != 4; j++) {
    mean[j].store(&mean_out[j * Vec::size]);
    M2[j].store(&m2_out[j * Vec::size]);
  }

  return rows;
}

static int64_t var128_v3_update(const float* data, float* mean_out, float* m2_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<float>;
  double running_mean = 0;
  double running_m2 = 0;
  int64_t running_n = 0;
  auto WIDTH = 32;

  for (int i = 0; i < rows; i += 256) {
    float meanf[4 * Vec::size];
    float cf[4 * Vec::size];
    float m2f[4 * Vec::size];
    int sub_rows = std::min(rows - i, int64_t(256));
    int64_t count = var128_v3(data + i * stride, meanf, m2f, cf, sub_rows, stride);

    double mean[4 * Vec::size];
    double m2[4 * Vec::size];
    for (int j = 0; j < WIDTH; j++) {
      mean[j] = meanf[j] - (double)cf[j];
      m2[j] = (double)m2f[j];
    }

    for (int step = 1; step != WIDTH; step = step << 1) {
      for (int j = 0; j < WIDTH; j += step * 2) {
        double delta = (mean[j] - mean[j + step]);
        mean[j] = (mean[j] + mean[j + step]) / 2;
        m2[j] += m2[j + step] + delta * delta * count / 2;
      }
      count *= 2;
    }

    auto delta = mean[0] - running_mean;
    running_mean  = (running_mean * running_n + count * mean[0]) / (count + running_n);
    auto delta2 = mean[0] - running_mean;
    running_m2 += m2[0] + (delta * delta2 * count * running_n) / (count + running_n);
    running_n += count;
    // std::cout << "running_mean: " << running_mean << " running_m2: " << (running_m2)/running_n << " " << delta << "\n";
  }
  *mean_out = (float)running_mean;
  *m2_out =  (float)running_m2;
  return running_n;
}

static int64_t var128_v4(const float* data, float* mean_out, float* m2_out, int64_t rows, int64_t stride) {
  float c = 0;
  float mean = 0;

  // First estimate the mean using Kahan summation
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 32; j++) {
      auto val = data[row * stride + j];
      auto y = val - c;
      auto t = mean + y;
      c = (t - mean) - y;
      mean = t;
    }
  }

  mean /= rows * 32;

  float M2 = 0;
  float correction = 0;
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 32; j++) {
      auto val = data[row * stride + j];
      auto delta = val - mean;
      M2 = delta * delta + M2;
      correction += delta;
    }
  }

  M2 -= correction * correction / (rows * 32);
  *mean_out = mean;
  *m2_out = M2;

  return rows * 32;
}


template <>
int64_t var128_v2(const float* data, double* mean_out, double* m2_out, int64_t rows, int64_t stride) {
  using Vec = Vec256<double>;

  for (int offset = 0; offset != 32; offset += 16) {
    Vec mean[4] = {0, 0, 0, 0};
    Vec M2[4] = {0, 0, 0, 0};

    HarmonicSequence<double> harmonic;

    auto update = [&](int j, double factor, const Vec& val) {
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
      floats.load(&data[row * stride + offset + floats.size]);
      update(2, factor, cast_double(floats.low()));
      update(3, factor, cast_double(floats.high()));
    }

    for (int j = 0; j < 4; j++) {
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

    auto out = res.data<float>();
    auto data = self.data<float>();
    auto numel = self.numel();
    if (!dim.has_value()) {
      *out = reduce_all(data, numel);
      return;
    }
  }

  static float reduce_all(const float* data, int size) {
    float mean[WIDTH];
    float M2[WIDTH];

    int64_t k = size / WIDTH;
    int64_t count = var128_v3_update(data, mean, M2, k, WIDTH);

    // for (int step = 1; step != WIDTH; step = step << 1) {
    //   for (int j = 0; j < WIDTH; j += step * 2) {
    //     auto delta = mean[j] - mean[j + step];
    //     mean[j] = (mean[j] + mean[j + step]) / 2;
    //     M2[j] += M2[j + step] + delta * delta * count / 2;
    //   }
    //   count *= 2;
    // }

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
