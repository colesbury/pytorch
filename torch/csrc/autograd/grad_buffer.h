#pragma once

#include <vector>
#include <utility>
#include <memory>

namespace thpp {
struct Tensor;
}

namespace torch { namespace autograd {

struct GradBuffer {
  explicit GradBuffer(size_t size);
  GradBuffer(GradBuffer&& other) = default;

  void addGrad(size_t idx, std::unique_ptr<thpp::Tensor> tensor);
  std::vector<std::unique_ptr<thpp::Tensor>> tensors();
  thpp::Tensor& operator[](size_t pos);
  const thpp::Tensor& operator[](size_t pos) const;

private:
  std::vector<std::pair<std::unique_ptr<thpp::Tensor>, bool>> buffer;
};

}}  // namespace torch::autograd
