#include "ATen/native/cpu/VarianceKernel.h"

#include <algorithm>
#include <numeric>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at {
namespace native {

using namespace vec256;

template<typename F>
static void parallel_for(int64_t end, int64_t step, bool parallelize, F func) {
  if (parallelize) {
    tbb::parallel_for<int64_t>(0, end, step, func);
  } else {
    for (int64_t i = 0; i < end; i += step) {
      func(i);
    }
  }
}

struct VecLoader {
  template<typename scalar_t>
  Vec256<scalar_t> load(const scalar_t* data, int j) const {
    using Vec = Vec256<scalar_t>;
    return Vec::s_load(&data[j * Vec::size]);
  }
};

struct PartialLoader {
  PartialLoader(int width) : width(width) {}

  template<typename scalar_t>
  Vec256<scalar_t> load(const scalar_t* data, int j) const {
    using Vec = Vec256<scalar_t>;
    int len = width - j * Vec::size;
    if (len >= Vec::size) {
      return Vec::s_load(&data[j * Vec::size]);
    } else if (len > 0) {
      Vec ret;
      ret.load_partial(&data[j * Vec::size], len);
      return ret;
    } else {
      return Vec(0);
    }
  }

  int width;
};

template<typename Loader>
static void var128(const float* data, double* sum_out, double* m2_out, int64_t rows, int64_t stride, const Loader& loader) {
  using Vec = Vec256<float>;
  Vec c[4] = {0, 0, 0, 0};
  Vec mean[4] = {0, 0, 0, 0};

  // First estimate the mean using Kahan summation
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 4; j++) {
      auto val = loader.load(&data[row * stride], j);
      auto y = val - c[j];
      auto t = mean[j] + y;
      c[j] = (t - mean[j]) - y;
      mean[j] = t;
    }
  }

  float inv_rows = 1.0f / rows;
  for (int j = 0; j != 4; j++) {
    auto low = cast_double(mean[j].low()) - cast_double(c[j].low());
    auto high = cast_double(mean[j].high()) - cast_double(c[j].high());
    low.store(&sum_out[j * Vec::size]);
    high.store(&sum_out[j * Vec::size + Vec::size / 2]);
    mean[j] *= inv_rows;
  }

  Vec M2[4] = {0, 0, 0, 0};
  Vec correction[4] = {0, 0, 0, 0};
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 4; j++) {
      auto val = loader.load(&data[row * stride], j);
      auto delta = val - mean[j];
      M2[j] = fmadd(delta, delta, M2[j]);
      correction[j] += delta;
    }
  }

  for (int j = 0; j != 4; j++) {
    M2[j] -= inv_rows * correction[j] * correction[j];
  }

  for (int j = 0; j != 4; j++) {
    auto low = cast_double(M2[j].low());
    low.store(&m2_out[j * Vec::size]);
    auto high = cast_double(M2[j].high());
    high.store(&m2_out[j * Vec::size + Vec::size / 2]);
  }
}

template<typename scalar_t>
struct VarReduction {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);

  using Vec = Vec256<scalar_t>;
  using vec4d = Vec256<double>;
  using range = tbb::blocked_range<int64_t>;

  struct VarOp {
    VarOp(const scalar_t* data, int64_t stride, int width=WIDTH)
     : data(data), stride(stride), width(width) {
      memset(sum, 0, sizeof(sum));
      memset(m2, 0, sizeof(m2));
    }
    VarOp(VarOp& s, tbb::split) : VarOp(s.data, s.stride, s.width) {
    }
    void operator()(const range& r) {
      if (n == 0 && r.size() <= 256) {
        if (__builtin_expect(width == WIDTH, 1)) {
          var128(&data[r.begin() * stride], sum, m2, r.size(), stride, VecLoader());
        } else {
          var128(&data[r.begin() * stride], sum, m2, r.size(), stride, PartialLoader(width));
        }
        n = r.size();
        return;
      }
      for (int64_t i = r.begin(); i < r.end(); i += 256) {
        int64_t end = std::min(r.end(), i + 256);
        auto op = VarOp(*this, tbb::split());
        op(range(i, end));
        join(op);
      }
    }
    void join(VarOp& rhs) {
      if (rhs.n == 0) {
        return;
      } else if (n == 0) {
        *this = rhs;
      } else {
        update(sum, m2, rhs.sum, rhs.m2, n, rhs.n);
        n += rhs.n;
      }
    }
    std::tuple<double, double> horizontal_reduce() {
      int64_t count = n;
      for (int step = 1; step != WIDTH; step = step << 1) {
        for (int j = 0; j < WIDTH; j += step * 2) {
          double delta = (sum[j] - sum[j + step]);
          sum[j] += sum[j + step];
          m2[j] += m2[j + step] + delta * delta / (2 * count);
        }
        count *= 2;
      }
      return std::make_tuple(sum[0] / count, m2[0]);
    }
    __at_align32__ double sum[WIDTH];
    __at_align32__ double m2[WIDTH];
    int64_t n = 0;
    const scalar_t* data;
    int64_t stride;
    int width;
  };

  static void apply(Tensor& res, const Tensor& self, at::optional<int64_t> dim) {
    internal::init_tbb_num_threads();

    auto out = res.data<float>();
    auto data = self.data<float>();
    auto numel = self.numel();
    if (!dim.has_value()) {
      *out = reduce_all(data, numel);
      return;
    }

    int64_t n = self.size(*dim);
    int64_t stride = self.stride(*dim);
    int64_t batch = numel / (n * stride);
    bool paralellize = batch * n > internal::TBB_GRAIN_SIZE;
    parallel_for(batch, 1, paralellize, [=](int64_t b) {
      if (stride == 1) {
        out[b] = reduce_all(&data[b * n], n);
      } else {
        reduce2d(&data[b * n * stride], &out[b * stride], n, stride, stride);
      }
    });
  }

  static float reduce_all(const float* data, int size) {
    double mean = 0;
    double m2 = 0;
    int64_t k = size / WIDTH;

    if (k > 0) {
      VarOp op(data, WIDTH);
      if (size > internal::TBB_GRAIN_SIZE / WIDTH) {
        tbb::parallel_deterministic_reduce(
            range(0, k, internal::TBB_GRAIN_SIZE / WIDTH),
            op);
      } else {
        op(range(0, k));
      }
      std::tie(mean, m2) = op.horizontal_reduce();
    }

    for (int i = k * WIDTH; i != size; i++) {
      double delta = data[i] - mean;
      mean += delta / (i + 1);
      double delta2 = data[i] - mean;
      m2 += delta * delta2;
    }

    return (float)(m2 / size);
  }

  static void reduce2d(const scalar_t* data, scalar_t* out, int64_t rows, int64_t cols, int64_t stride) {
    bool paralellize = cols * rows > internal::TBB_GRAIN_SIZE;
    parallel_for(cols, WIDTH, paralellize, [=](int64_t col) {
      int width = std::min(WIDTH, (int)(cols - col));
      VarOp op(&data[col], stride, width);
      op(range(0, rows));
      for (int i = 0; i != width; i++) {
        out[col + i] = op.m2[i] / op.n;
      }
    });
  }

  static void update(double* sum_a, double* m2_a, const double* sum_b, const double* m2_b, int64_t n_a, int64_t n_b) {
    double ratio = n_b / (double)n_a;
    double scale = n_a / (double)(n_b * (n_a + n_b));
    auto update_vec = [&](vec4d& sum_a, vec4d& m2_a, const vec4d& sum_b, const vec4d& m2_b) {
      auto delta = ratio * sum_a - sum_b;
      sum_a += sum_b;
      m2_a += m2_b + (delta * delta) * scale;
    };
    for (int i = 0; i != WIDTH; i += vec4d::size) {
      auto sum1 = vec4d::s_load(&sum_a[i]);
      auto sum2 = vec4d::s_load(&sum_b[i]);
      auto m2_1 = vec4d::s_load(&m2_a[i]);
      auto m2_2 = vec4d::s_load(&m2_b[i]);
      update_vec(sum1, m2_1, sum2, m2_2);
      sum1.store(&sum_a[i]);
      m2_1.store(&m2_a[i]);
    }
  }
};

static void var_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim, bool unbiased) {
  // AT_DISPATCH_FLOATING_TYPES(self.type(), "var", [&] {
    VarReduction<float>::apply(result, self, dim);
  // });
}

REGISTER_DISPATCH(var_kernel, &var_kernel_impl);

}}
