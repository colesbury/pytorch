#pragma once

#include <vector>
#include <utility>
#include <memory>

namespace thpp {
struct Tensor;
}
struct THVariable;

namespace torch { namespace autograd {

struct GradBuffer {
  explicit GradBuffer(size_t size);
  GradBuffer(GradBuffer&& other) = default;

  void addGrad(size_t idx, std::shared_ptr<THVariable>&& var);
  static std::vector<std::shared_ptr<THVariable>> variables(GradBuffer&& buffer);

private:
  std::vector<std::pair<std::unique_ptr<thpp::Tensor>, bool>> buffer;
};

}}  // namespace torch::autograd
